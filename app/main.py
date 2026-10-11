import base64
import binascii
import io
import os
import threading
import time
import uuid
from collections import Counter, defaultdict

import cv2
import numpy as np
from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for
from flask_login import current_user, login_required
from PIL import Image
from werkzeug.utils import secure_filename

from . import db
from .models import Prediction
from .predictor import LEAF_CONFIDENCE_THRESHOLD, build_result_explanation, predictor

bp = Blueprint("main", __name__)
_jobs = {}
_jobs_lock = threading.RLock()
_JOB_TTL_SECONDS = 60 * 60
_STEP_NAMES = (
    ("Step 1 — Image received & validated", "Checking the uploaded image and validating the file..."),
    ("Step 2 — Preprocessing", "Preparing the leaf image for the model..."),
    ("Step 3 — Model inference", "Running the trained CNN and extracting disease probabilities..."),
    ("Step 4 — Grad-CAM", "Generating a Grad-CAM heatmap to show where the model focused..."),
    ("Step 5 — Result assembly", "Preparing your diagnosis, confidence, severity, and recommendations..."),
)


def valid_image(raw, filename="image.jpg"):
    if len(raw) > current_app.config["MAX_CONTENT_LENGTH"]:
        raise ValueError("Image must be smaller than 8 MB.")
    if not filename.lower().endswith(tuple(current_app.config["UPLOAD_EXTENSIONS"])):
        raise ValueError("Only JPEG and PNG images are accepted.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        if image.format not in ("JPEG", "PNG"):
            raise ValueError("Only JPEG and PNG images are accepted.")
    except Exception as exc:
        raise ValueError("The uploaded file is not a valid JPEG or PNG.") from exc


def save_visuals(raw, heatmap, user_id):
    folder = os.path.join(current_app.instance_path, "uploads", str(user_id))
    os.makedirs(folder, exist_ok=True)
    stem = uuid.uuid4().hex
    image_name = stem + ".jpg"
    heatmap_name = stem + "_heatmap.jpg"
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded image could not be decoded.")
    cam = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(image, 0.55, cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET), 0.45, 0)
    if not cv2.imwrite(os.path.join(folder, image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise ValueError("The uploaded image could not be saved.")
    if not cv2.imwrite(os.path.join(folder, heatmap_name), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise ValueError("The analysis image could not be saved.")
    return os.path.join(str(user_id), image_name), os.path.join(str(user_id), heatmap_name)


def _update_job(job_id, **changes):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job.update(changes)


def _update_step(job_id, step_index, status):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job["steps"][step_index]["status"] = status


def _job_response(job):
    """Expose a small JSON-safe job snapshot without returning uploaded image bytes."""
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "steps": job["steps"],
        "result": job.get("result"),
        "error": job.get("error"),
    }


def _validate_leaf_confidence(result):
    # Reject low-confidence inputs to avoid presenting unreliable diagnoses for unrelated images.
    if result["max_confidence"] < LEAF_CONFIDENCE_THRESHOLD:
        raise ValueError(
            "This doesn't look like a leaf image our model recognizes — please upload a clear photo of a single plant leaf."
        )


def _prune_jobs():
    """Bound in-process job state while retaining active jobs for their pollers."""
    cutoff = time.time() - _JOB_TTL_SECONDS
    with _jobs_lock:
        expired = [key for key, job in _jobs.items() if job["created"] < cutoff and job["status"] in ("complete", "failed")]
        for key in expired:
            _jobs.pop(key, None)
        completed = sorted(
            ((key, job["created"]) for key, job in _jobs.items() if job["status"] in ("complete", "failed")),
            key=lambda item: item[1],
        )
        for key, _ in completed[:max(0, len(_jobs) - 500)]:
            _jobs.pop(key, None)


def _process_job(app, job_id, user_id, raw, filename):
    """Run validation, model work, and one database insert outside the request thread."""
    with app.app_context():
        try:
            _update_job(job_id, status="running")
            _update_step(job_id, 0, "in_progress")
            if not raw:
                raise ValueError("Choose an image or capture one with your camera.")
            valid_image(raw, filename)
            _update_step(job_id, 0, "done")
            try:
                result = predictor.predict(raw, lambda index, state: _update_step(job_id, index, state))
            except Exception:
                current_app.logger.exception("Model inference failed for job %s", job_id)
                _mark_job_failed(job_id, "Model inference failed. Please retry the diagnosis.")
                return
            _update_step(job_id, 4, "in_progress")
            _validate_leaf_confidence(result)
            image_path, heatmap_path = save_visuals(raw, result["heatmap"], user_id)
            record = Prediction(
                user_id=user_id,
                filename=filename,
                image_path=image_path,
                heatmap_path=heatmap_path,
                label=result["label"],
                category=result["category"],
                confidence=result["confidence"],
                severity=result["severity"],
                treatment=result["treatment"],
                prevention=result["prevention"],
            )
            db.session.add(record)
            db.session.commit()
            with app.test_request_context():
                response = {
                    "id": record.id,
                    "label": result["label"],
                    "category": result["category"],
                    "confidence": result["confidence"],
                    "severity": result["severity"],
                    "treatment": result["treatment"],
                    "prevention": result["prevention"],
                    "attention_percentage": result["attention_percentage"],
                    "image_url": url_for("main.user_upload", path=image_path),
                    "heatmap_url": url_for("main.user_upload", path=heatmap_path),
                    "report_url": url_for("main.report", prediction_id=record.id),
                    "result_url": url_for(
                        "main.view_result",
                        prediction_id=record.id,
                        attention_percentage=result["attention_percentage"],
                    ),
                }
            _update_step(job_id, 4, "done")
            _update_job(job_id, status="complete", result=response)
        except (ValueError, RuntimeError) as exc:
            db.session.rollback()
            message = str(exc)
            _mark_job_failed(job_id, message)
        except Exception:
            db.session.rollback()
            current_app.logger.exception("Diagnosis result assembly failed for job %s", job_id)
            _mark_job_failed(job_id, "Analysis could not be completed. Please try again.")
        finally:
            db.session.remove()


def _mark_job_failed(job_id, message):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        for step in job["steps"]:
            if step["status"] == "in_progress":
                step["status"] = "failed"
                break
        job["status"] = "failed"
        job["error"] = message


def _request_image():
    """Read a multipart image or decode a browser camera data URL."""
    payload = request.get_json(silent=True) or {}
    uploaded = request.files.get("file")
    raw = uploaded.read() if uploaded else None
    filename = secure_filename(uploaded.filename) if uploaded else "camera.jpg"
    camera_data = request.form.get("camera_data") or payload.get("camera_data")
    if not raw and camera_data:
        try:
            raw = base64.b64decode(camera_data.split(",", 1)[-1], validate=True)
        except (binascii.Error, ValueError):
            raw = None
    return raw or b"", filename


@bp.get("/")
def index():
    return render_template("home.html")


@bp.route("/predict", methods=["GET", "POST"])
@bp.route("/diagnose", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "GET":
        return render_template("predict.html", input_size=predictor.input_size)
    raw, filename = _request_image()
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "user_id": current_user.id,
        "status": "queued",
        "steps": [{"name": name, "detail": detail, "status": "pending"} for name, detail in _STEP_NAMES],
        "created": time.time(),
    }
    _prune_jobs()
    with _jobs_lock:
        _jobs[job_id] = job
    app = current_app._get_current_object()
    worker = threading.Thread(target=_process_job, args=(app, job_id, current_user.id, raw, filename), daemon=True)
    worker.start()
    if request.accept_mimetypes.best == "application/json":
        return jsonify({"job_id": job_id, "status_url": url_for("main.predict_status", job_id=job_id)}), 202
    return render_template("progress.html", job_id=job_id, status_url=url_for("main.predict_status", job_id=job_id)), 202


@bp.get("/predict/status/<job_id>")
@login_required
def predict_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job or job["user_id"] != current_user.id:
            abort(404)
        return jsonify(_job_response(job))


@bp.post("/api/predict")
@login_required
def api_predict():
    raw, filename = _request_image()
    try:
        if not raw:
            raise ValueError("Choose an image or capture one with your camera.")
        valid_image(raw, filename)
        try:
            result = predictor.predict(raw)
        except Exception:
            db.session.rollback()
            current_app.logger.exception("Synchronous model inference failed for user %s", current_user.get_id())
            return jsonify(error="Model inference failed. Please retry the diagnosis."), 500
        _validate_leaf_confidence(result)
        image_path, heatmap_path = save_visuals(raw, result["heatmap"], current_user.id)
        record = Prediction(
            user_id=current_user.id, filename=filename, image_path=image_path, heatmap_path=heatmap_path,
            label=result["label"], category=result["category"], confidence=result["confidence"],
            severity=result["severity"], treatment=result["treatment"], prevention=result["prevention"],
        )
        db.session.add(record)
        db.session.commit()
        return jsonify({
            "id": record.id,
            "label": result["label"],
            "category": result["category"],
            "confidence": result["confidence"],
            "severity": result["severity"],
            "treatment": result["treatment"],
            "prevention": result["prevention"],
            "attention_percentage": result["attention_percentage"],
            "image_url": url_for("main.user_upload", path=image_path),
            "heatmap_url": url_for("main.user_upload", path=heatmap_path),
            "report_url": url_for("main.report", prediction_id=record.id),
            "result_url": url_for(
                "main.view_result",
                prediction_id=record.id,
                attention_percentage=result["attention_percentage"],
            ),
        })
    except (ValueError, RuntimeError) as exc:
        db.session.rollback()
        return jsonify(error=str(exc)), 400
    except Exception:
        db.session.rollback()
        current_app.logger.exception("Synchronous diagnosis result assembly failed for user %s", current_user.get_id())
        return jsonify(error="Model inference failed. Please retry the diagnosis."), 500


@bp.get("/history")
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", predictions=predictions)


@bp.get("/profile")
@login_required
def profile():
    name = current_user.email.partition("@")[0].replace(".", " ").replace("_", " ").replace("-", " ").title()
    return render_template("profile.html", name=name)


@bp.post("/history/<int:prediction_id>/delete")
@login_required
def delete_prediction(prediction_id):
    prediction = Prediction.query.filter_by(
        id=prediction_id,
        user_id=current_user.id,
    ).first_or_404()
    db.session.delete(prediction)
    db.session.commit()
    flash("Assessment deleted.", "success")
    return redirect(url_for("main.history"))


@bp.get("/result/<int:prediction_id>")
@login_required
def view_result(prediction_id):
    record = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first_or_404()
    attention_percentage = request.args.get("attention_percentage", type=float)
    if attention_percentage is not None and 0 <= attention_percentage <= 100:
        explanation = build_result_explanation(record.confidence, attention_percentage)
    else:
        explanation = (
            f"The model recorded {record.confidence:.1%} confidence for this saved result. "
            "Grad-CAM attention coverage was not retained for this earlier assessment."
        )
    result = {
        "id": record.id,
        "label": record.label,
        "category": record.category,
        "confidence": record.confidence,
        "severity": record.severity,
        "explanation": explanation,
        "treatment": record.treatment,
        "prevention": record.prevention,
        "image_url": url_for("main.user_upload", path=record.image_path),
        "heatmap_url": url_for("main.user_upload", path=record.heatmap_path),
        "report_url": url_for("main.report", prediction_id=record.id),
    }
    return render_template("result.html", result=result)


@bp.get("/dashboard")
@login_required
def dashboard():
    """Aggregate only the signed-in user's records for the dashboard charts."""
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.asc()).all()
    total = len(predictions)
    healthy_count = sum(1 for prediction in predictions if prediction.category == "Healthy Leaf")
    diseased_count = total - healthy_count
    crop_counter = Counter()
    disease_counter = Counter()
    confidence_by_day = defaultdict(list)
    severity_counter = Counter()

    for prediction in predictions:
        label = prediction.label or "Unknown"
        if " — " in label:
            crop_name, disease_name = label.split(" — ", 1)
            crop_counter[crop_name.strip()] += 1
            if prediction.category != "Healthy Leaf":
                disease_counter[disease_name.strip()] += 1
        else:
            crop_counter["Unknown"] += 1
            if prediction.category != "Healthy Leaf":
                disease_counter[label.strip()] += 1

        confidence_value = float(prediction.confidence or 0)
        if confidence_value > 1:
            confidence_value = confidence_value / 100.0
        confidence_by_day[prediction.created_at.date().isoformat()].append(confidence_value * 100.0)
        severity_counter[str(prediction.severity or "Unknown").strip()] += 1

    ordered_severity = []
    for name in ("Mild", "Moderate", "Severe"):
        if name in severity_counter:
            ordered_severity.append(name)
    for name in severity_counter:
        if name not in ordered_severity:
            ordered_severity.append(name)

    confidence_dates = sorted(confidence_by_day)
    confidence_values = [
        round(sum(confidence_by_day[day]) / len(confidence_by_day[day]), 1)
        for day in confidence_dates
    ]

    dashboard_data = {
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "disease_labels": [name for name, _ in disease_counter.most_common(6)],
        "disease_counts": [count for _, count in disease_counter.most_common(6)],
        "confidence_dates": confidence_dates,
        "confidence_values": confidence_values,
        "severity_labels": ordered_severity,
        "severity_counts": [severity_counter.get(name, 0) for name in ordered_severity],
    }

    return render_template(
        "dashboard.html",
        total=total,
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        healthy_ratio=round(healthy_count / total * 100, 1) if total else 0,
        disease_ratio=round(diseased_count / total * 100, 1) if total else 0,
        average_confidence=round(sum(float(prediction.confidence or 0) for prediction in predictions) * 100 / total, 1) if total else 0,
        dashboard_data=dashboard_data,
    )


def _owned_image_path(record, relative_path):
    if not relative_path:
        return None
    root = os.path.realpath(os.path.join(current_app.instance_path, "uploads", str(record.user_id)))
    path = os.path.realpath(os.path.join(root, os.path.basename(relative_path)))
    try:
        if os.path.commonpath((root, path)) != root or not os.path.isfile(path):
            return None
    except (OSError, ValueError):
        return None
    return path


@bp.get("/report/<int:prediction_id>")
@login_required
def report(prediction_id):
    """Generate a PDF only after confirming the requested assessment belongs to the user."""
    record = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first_or_404()
    try:
        images = {
            "original-image": _owned_image_path(record, record.image_path),
            "heatmap-image": _owned_image_path(record, record.heatmap_path),
        }
        label = record.label or "Diagnosis"
        crop_name = label.split(" — ", 1)[0] if " — " in label else None
        report_label = label.replace(" — ", " - ")
        generated_at = time.strftime("%d/%m/%Y %H:%M UTC", time.gmtime())
        html = render_template(
            "report.html",
            prediction=record,
            report_label=report_label,
            images=images,
            crop_name=crop_name,
            health_status=(
                "Healthy" if record.category == "Healthy Leaf"
                else "Diseased" if record.category
                else "Not available"
            ),
            generated_at=generated_at,
        )
        from xhtml2pdf import pisa
        output = io.BytesIO()
        pdf_result = pisa.CreatePDF(html, dest=output, link_callback=lambda uri, rel: images.get(uri))
        if pdf_result.err or not output.getvalue():
            raise RuntimeError("PDF renderer rejected the report.")
        output.seek(0)
        return send_file(output, mimetype="application/pdf", as_attachment=True, download_name=f"leafguard-report-{record.id}.pdf")
    except Exception:
        current_app.logger.exception("Failed to generate diagnosis report for prediction %s", record.id)
        return "The report could not be generated. Please try again later.", 503


@bp.get("/uploads/<path:path>")
@login_required
def user_upload(path):
    # Database paths contain a user prefix; serve only the current user's basename.
    requested = os.path.basename(os.path.normpath(path))
    return send_from_directory(os.path.join(current_app.instance_path, "uploads", str(current_user.id)), requested)
