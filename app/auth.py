from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, current_user
from . import db
from .models import User

bp = Blueprint("auth", __name__, url_prefix="/auth")
@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated: return redirect(url_for("main.index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower(); password = request.form.get("password", "")
        if not email or len(password) < 8: flash("Use a valid email and password of at least 8 characters.", "danger")
        elif User.query.filter_by(email=email).first(): flash("That email is already registered.", "warning")
        else:
            user = User(email=email); user.set_password(password); db.session.add(user); db.session.commit()
            login_user(user); return redirect(url_for("main.index"))
    return render_template("auth.html", title="Create account", register=True)
@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(email=request.form.get("email", "").strip().lower()).first()
        if user and user.check_password(request.form.get("password", "")): login_user(user); return redirect(url_for("main.index"))
        flash("Email or password is incorrect.", "danger")
    return render_template("auth.html", title="Sign in", register=False)
@bp.get("/logout")
def logout(): logout_user(); return redirect(url_for("main.index"))
