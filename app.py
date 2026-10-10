import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify

from config import Config
from views.api_view import api_bp
from views.page_view import pages_bp
from services.bootstrap import build_services


def create_app() -> Flask:
    load_dotenv(override=True)
    yolo_cfg_dir = Path(__file__).resolve().parent / ".yolo_config"
    yolo_cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_cfg_dir))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app = Flask(__name__, static_folder="static", template_folder="templates")
    cfg = Config()
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_content_length
    app.config["APP_CONFIG"] = cfg

    build_services(app, cfg)
    app.register_blueprint(api_bp)
    app.register_blueprint(pages_bp)
    app.logger.setLevel(logging.INFO)
    app.logger.info("application booted")

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "Uploaded file too large."}), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
