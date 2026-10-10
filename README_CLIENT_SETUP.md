# No Helmet Detection - Client Setup

## Required Python

Install **Python 3.9.10** on the client PC. Any Python 3.9.x should work, but this package was built and tested with Python 3.9.10 on Windows.

Download Python 3.9.10:
https://www.python.org/downloads/release/python-3910/

During install, enable **Add Python to PATH**.

## Setup

1. Extract the zip file.
2. Double-click `setup_client.cmd`.
3. Wait for dependency installation and model warmup to finish.

The setup script creates `venv`, installs all Python packages, creates `.env`, initializes the database, verifies YOLO model files, and downloads/warms PaddleOCR runtime models.

## Run

Double-click:

```cmd
run_app.cmd
```

Then open:

```text
http://127.0.0.1:5000/
```

## Configure Email

Email sending is optional. To enable it, edit `.env` after setup:

```text
SMTP_USERNAME=
SMTP_PASSWORD=
SENDER_EMAIL=
AUTO_EMAIL_RECIPIENT=
```

Use an app password for Gmail or the SMTP provider.

## Notes

- Internet is required during setup for Python packages and PaddleOCR model downloads.
- The custom YOLO model files are included in the `models` folder.
- If setup fails halfway, run `setup_client.cmd` again.
