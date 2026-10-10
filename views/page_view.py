from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)


@pages_bp.get("/", endpoint="home")
def home_page():
    return render_template("home.html")


@pages_bp.get("/image-detection", endpoint="image")
def image_page():
    return render_template("image_detection.html")


@pages_bp.get("/video-detection", endpoint="video")
def video_page():
    return render_template("video_detection.html")


@pages_bp.get("/live-camera", endpoint="live")
def live_page():
    return render_template("live_camera.html")


@pages_bp.get("/violations-dashboard", endpoint="dashboard")
def dashboard_page():
    return render_template("violations_dashboard.html")


@pages_bp.get("/email-page", endpoint="email")
def email_page():
    return render_template("email.html")


@pages_bp.get("/about", endpoint="about")
def about_page():
    return render_template("about.html")

