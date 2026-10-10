import smtplib
import unittest
from unittest.mock import patch

from services.email_service import EmailService


class EmailServiceTestCase(unittest.TestCase):
    def test_password_whitespace_is_removed(self):
        service = EmailService(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username="user@gmail.com",
            password="abcd efgh\nijkl\tmnop",
            sender_email="user@gmail.com",
            use_tls=True,
        )
        self.assertEqual(service.password, "abcdefghijklmnop")

    def test_gmail_auth_error_has_actionable_message(self):
        service = EmailService(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username="user@gmail.com",
            password="bad password",
            sender_email="user@gmail.com",
            use_tls=True,
        )

        class _FakeSMTP:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def ehlo(self):
                return None

            def starttls(self):
                return None

            def login(self, username, password):
                raise smtplib.SMTPAuthenticationError(535, b"BadCredentials")

        with patch("smtplib.SMTP", _FakeSMTP):
            with self.assertRaisesRegex(RuntimeError, "Google App Password"):
                service.send_violation_email(
                    to_email="to@example.com",
                    plate_number="KA01AB1234",
                    image_path="missing.jpg",
                    timestamp="now",
                )


if __name__ == "__main__":
    unittest.main()
