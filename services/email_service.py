from __future__ import annotations

import logging
import mimetypes
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Dict

LOGGER = logging.getLogger(__name__)


class EmailService:
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        sender_email: str,
        use_tls: bool = True,
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username.strip()
        self.password = "".join((password or "").split())
        self.sender_email = (sender_email or username).strip()
        self.use_tls = use_tls

    def _gmail_auth_help(self) -> str:
        return (
            "Gmail rejected the SMTP login. Use your full Gmail address in SMTP_USERNAME/SENDER_EMAIL "
            "and a fresh 16-character Google App Password in SMTP_PASSWORD. Do not use your normal Gmail "
            "account password. App Passwords require 2-Step Verification to be enabled on the Google account."
        )

    def is_configured(self) -> bool:
        if "YOUR_GMAIL_ADDRESS" in self.username or "YOUR_GMAIL_ADDRESS" in self.sender_email:
            return False
        if "your-16-character" in self.password.lower():
            return False
        return bool(self.smtp_server and self.smtp_port and self.username and self.password and self.sender_email)

    def config_status(self) -> Dict:
        password_without_spaces = self.password.replace(" ", "")
        return {
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port,
            "username": self.username,
            "sender_email": self.sender_email,
            "use_tls": self.use_tls,
            "configured": self.is_configured(),
            "password_length": len(password_without_spaces),
            "password_format_ok": len(password_without_spaces) == 16,
            "gmail_sender_match": (
                self.smtp_server.lower() != "smtp.gmail.com"
                or self.username.lower() == self.sender_email.lower()
            ),
        }

    def send_violation_email(
        self,
        to_email: str,
        plate_number: str,
        image_path: str,
        timestamp: str,
    ) -> None:
        if not self.is_configured():
            raise RuntimeError(
                "SMTP is not configured. Set your real Gmail address in SMTP_USERNAME and SENDER_EMAIL, "
                "and keep the Gmail app password in SMTP_PASSWORD."
            )
        if not to_email:
            raise ValueError("Recipient email is required.")

        subject = f"No Helmet Violation Detected: {plate_number or 'UNKNOWN'}"
        body = (
            "Traffic Violation Alert\n\n"
            f"Plate Number: {plate_number or 'UNREADABLE'}\n"
            f"Timestamp: {timestamp}\n"
            "Violation Type: Rider without helmet\n"
        )
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.sender_email
        message["To"] = to_email
        message.set_content(body)

        path = Path(image_path)
        if path.exists():
            mime_type, _ = mimetypes.guess_type(str(path))
            main_type, sub_type = (mime_type or "image/jpeg").split("/", 1)
            with path.open("rb") as f:
                message.add_attachment(
                    f.read(),
                    maintype=main_type,
                    subtype=sub_type,
                    filename=path.name,
                )

        LOGGER.info("sending email to=%s plate=%s via=%s:%s", to_email, plate_number, self.smtp_server, self.smtp_port)
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.ehlo()
                if self.use_tls:
                    server.starttls()
                    server.ehlo()
                server.login(self.username, self.password)
                server.send_message(message)
        except smtplib.SMTPAuthenticationError as exc:
            LOGGER.exception("SMTP authentication failed")
            raise RuntimeError(f"SMTP authentication failed: {self._gmail_auth_help()}") from exc
        except smtplib.SMTPException as exc:
            LOGGER.exception("SMTP send failed")
            raise RuntimeError(f"SMTP send failed: {exc}") from exc
