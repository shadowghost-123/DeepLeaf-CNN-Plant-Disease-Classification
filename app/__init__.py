import os
from datetime import datetime
from pathlib import Path

from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

PROJECT_ROOT = Path(__file__).resolve().parent.parent

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message_category = "warning"


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-change-me"),
        SQLALCHEMY_DATABASE_URI=os.environ.get(
            "DATABASE_URL", "sqlite:///" + os.path.join(app.instance_path, "plantdoctor.sqlite")
        ),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        UPLOAD_EXTENSIONS={".jpg", ".jpeg", ".png"},
        MODEL_PATH=str(PROJECT_ROOT / "models" / "Finetune_Epoch 04.keras"),
        CLASS_INDICES_PATH=str(PROJECT_ROOT / "models" / "Final_class_indices.json"),
        LOAD_MODEL=True,
    )
    if test_config:
        app.config.update(test_config)
    os.makedirs(app.instance_path, exist_ok=True)
    db.init_app(app)
    login_manager.init_app(app)
    from .models import User, Prediction
    with app.app_context():
        db.create_all()
        # Firebase was intentionally replaced with local SQLite: it is simple to deploy,
        # keeps farmers' images/history private, and has no recurring service cost.
        # This also avoids a billing card in 2026 while this zero-cost project is evaluated.
        from .predictor import predictor
        if app.config["LOAD_MODEL"]:
            predictor.configure(app.config["MODEL_PATH"], app.config["CLASS_INDICES_PATH"])
            model_path = Path(app.config["MODEL_PATH"])
            timestamp_path = model_path / "model.weights.h5" if model_path.is_dir() else model_path
            last_modified = datetime.fromtimestamp(
                timestamp_path.stat().st_mtime
            ).astimezone().isoformat(timespec="seconds")
            print(
                f"Loaded model: path={model_path}, "
                f"last_modified={last_modified}",
                flush=True,
            )
    from .auth import bp as auth_bp
    from .main import bp as main_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    return app
