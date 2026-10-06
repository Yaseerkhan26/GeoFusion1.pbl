"""
===============================================================================
File: tests/test_auth.py
Purpose: Unit Tests for GeoFusion AI Authentication Module
===============================================================================
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.auth import (
    authenticate_user,
    check_password_strength,
    hash_password,
    register_user,
    validate_email,
    verify_password,
)


class TestAuthModule(unittest.TestCase):
    def setUp(self):
        """Create a temporary user database for clean isolation."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_users_file = Path(self.temp_dir.name) / "users.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_email_validation(self):
        self.assertTrue(validate_email("researcher@geofusion.ai"))
        self.assertTrue(validate_email("user.name+tag@domain.co.uk"))
        self.assertFalse(validate_email("invalid-email"))
        self.assertFalse(validate_email("@domain.com"))
        self.assertFalse(validate_email("user@.com"))
        self.assertFalse(validate_email(""))

    def test_password_hashing_and_verification(self):
        salt = os.urandom(16).hex()
        password = "SecurePassword123!"
        hashed = hash_password(password, salt)

        self.assertNotEqual(password, hashed)
        self.assertTrue(verify_password(password, hashed, salt))
        self.assertFalse(verify_password("WrongPassword123!", hashed, salt))

    def test_password_strength_evaluator(self):
        weak = check_password_strength("short")
        self.assertFalse(weak["is_acceptable"])
        self.assertEqual(weak["label"], "Weak")

        strong = check_password_strength("GeoFusion@2026!Secure")
        self.assertTrue(strong["is_acceptable"])
        self.assertEqual(strong["label"], "Strong")

    def test_user_registration_and_authentication(self):
        with patch("src.utils.auth.USERS_FILE", self.temp_users_file):
            # Register new user
            success, msg = register_user(
                full_name="Dr. Jane Doe",
                email="jane.doe@geofusion.ai",
                password="Password@2026",
                confirm_password="Password@2026",
            )
            self.assertTrue(success, msg)

            # Mismatched password
            success_mismatch, msg_mismatch = register_user(
                full_name="Dr. Jane Doe",
                email="jane2@geofusion.ai",
                password="Password@2026",
                confirm_password="DifferentPassword@2026",
            )
            self.assertFalse(success_mismatch)
            self.assertIn("do not match", msg_mismatch)

            # Authenticate valid user
            auth_ok, user_info, auth_msg = authenticate_user("jane.doe@geofusion.ai", "Password@2026")
            self.assertTrue(auth_ok)
            self.assertEqual(user_info["full_name"], "Dr. Jane Doe")
            self.assertEqual(user_info["email"], "jane.doe@geofusion.ai")

    def test_password_reset_and_lockout(self):
        with patch("src.utils.auth.USERS_FILE", self.temp_users_file):
            # Register new user
            register_user(
                full_name="Dr. Jane Doe",
                email="jane.doe@geofusion.ai",
                password="Password@2026",
                confirm_password="Password@2026",
            )

            # Test successful password reset
            from src.utils.auth import reset_password
            res_ok, msg_ok = reset_password(
                email="jane.doe@geofusion.ai",
                new_password="NewSecurePassword@2026",
                confirm_password="NewSecurePassword@2026",
            )
            self.assertTrue(res_ok, msg_ok)

            # Authenticate with new password
            auth_ok, _, _ = authenticate_user("jane.doe@geofusion.ai", "NewSecurePassword@2026")
            self.assertTrue(auth_ok)

            # Authenticate with old password should fail
            auth_old, _, _ = authenticate_user("jane.doe@geofusion.ai", "Password@2026")
            self.assertFalse(auth_old)

            # Test account lockout after 5 consecutive failed login attempts
            for i in range(5):
                authenticate_user("jane.doe@geofusion.ai", "WrongPass123!")

            # 6th attempt should return lockout message
            auth_locked, _, lock_msg = authenticate_user("jane.doe@geofusion.ai", "NewSecurePassword@2026")
            self.assertFalse(auth_locked)
            self.assertIn("locked", lock_msg.lower())


if __name__ == "__main__":
    unittest.main()

