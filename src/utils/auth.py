"""
===============================================================================
File: src/utils/auth.py
Purpose: Production-Grade Authentication, Password Hashing, User Management & Session Security
===============================================================================
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Base Directory & User Store Location
BASE_DIR = Path(__file__).resolve().parent.parent.parent
USERS_FILE = BASE_DIR / "data" / "users.json"
SECRET_KEY = os.environ.get("GEOFUSION_SECRET_KEY") or os.urandom(32).hex()

# PBKDF2 Configuration
HASH_ALGORITHM = "sha256"
ITERATIONS = 100_000
SALT_SIZE = 16


def _ensure_users_file():
    """Ensures the users JSON data store exists with an optional admin user if environment variables are provided."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists() or USERS_FILE.stat().st_size == 0:
        admin_email = os.environ.get("GEOFUSION_ADMIN_EMAIL")
        admin_password = os.environ.get("GEOFUSION_ADMIN_PASSWORD")
        default_users = {}
        if admin_email and admin_password:
            clean_email = admin_email.strip().lower()
            salt = os.urandom(SALT_SIZE).hex()
            password_hash = hash_password(admin_password, salt)
            default_users[clean_email] = {
                "full_name": "GeoFusion Admin",
                "email": clean_email,
                "password_hash": password_hash,
                "salt": salt,
                "role": "admin",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=4)


def _load_users() -> Dict[str, Dict[str, Any]]:
    """Loads all registered users from storage."""
    _ensure_users_file()
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: Dict[str, Dict[str, Any]]) -> bool:
    """Saves user dictionary to storage."""
    try:
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=4)
        return True
    except Exception:
        return False


def hash_password(password: str, salt: str) -> str:
    """Hashes a password using PBKDF2-HMAC-SHA256 with a unique salt."""
    key = hashlib.pbkdf2_hmac(
        HASH_ALGORITHM,
        password.encode("utf-8"),
        bytes.fromhex(salt),
        ITERATIONS,
    )
    return key.hex()


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a plain password against the stored hash and salt."""
    computed_hash = hash_password(password, salt)
    return hashlib.sha256(computed_hash.encode()).hexdigest() == hashlib.sha256(stored_hash.encode()).hexdigest()


def validate_email(email: str) -> bool:
    """Validates email format using regex."""
    if not email or len(email) > 254:
        return False
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


def check_password_strength(password: str) -> Dict[str, Any]:
    """
    Evaluates password strength and returns score, label, and missing requirements.
    Score ranges from 0 to 4.
    """
    missing = []
    if len(password) < 8:
        missing.append("At least 8 characters")
    if not re.search(r"[A-Z]", password):
        missing.append("At least one uppercase letter (A-Z)")
    if not re.search(r"[a-z]", password):
        missing.append("At least one lowercase letter (a-z)")
    if not re.search(r"[0-9]", password):
        missing.append("At least one digit (0-9)")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        missing.append("At least one special character (!@#$%^&*)")

    score = 5 - len(missing)

    if score <= 1:
        label = "Weak"
        color = "#EF4444"
    elif score == 2 or score == 3:
        label = "Fair"
        color = "#F59E0B"
    elif score == 4:
        label = "Good"
        color = "#3B82F6"
    else:
        label = "Strong"
        color = "#10B981"

    return {
        "score": score,
        "max_score": 5,
        "label": label,
        "color": color,
        "missing": missing,
        "is_acceptable": score >= 3 and len(password) >= 8,
    }


def register_user(full_name: str, email: str, password: str, confirm_password: str) -> Tuple[bool, str]:
    """Registers a new user after strict validation."""
    clean_name = full_name.strip()
    clean_email = email.strip().lower()

    if not clean_name:
        return False, "Full Name is required."
    if not validate_email(clean_email):
        return False, "Please enter a valid email address."
    if not password:
        return False, "Password is required."
    if password != confirm_password:
        return False, "Passwords do not match."

    pw_eval = check_password_strength(password)
    if not pw_eval["is_acceptable"]:
        return False, f"Password is too weak: {', '.join(pw_eval['missing'])}"

    users = _load_users()
    if clean_email in users:
        return False, "An account with this email already exists."

    salt = os.urandom(SALT_SIZE).hex()
    password_hash = hash_password(password, salt)

    users[clean_email] = {
        "full_name": clean_name,
        "email": clean_email,
        "password_hash": password_hash,
        "salt": salt,
        "role": "researcher",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if _save_users(users):
        return True, "Registration successful! You can now log in."
    else:
        return False, "Failed to save user account to database."


def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Authenticates user credentials against stored hashes."""
    clean_email = email.strip().lower()

    if not clean_email or not password:
        return False, None, "Email and password are required."

    users = _load_users()
    if clean_email not in users:
        return False, None, "Invalid email or password."

    user_record = users[clean_email]
    
    # Check for temporary lock out after repeated failed login attempts
    failed_attempts = user_record.get("failed_attempts", 0)
    last_failed_time = user_record.get("last_failed_time", 0)
    current_time = time.time()
    
    if failed_attempts >= 5 and (current_time - last_failed_time) < 300: # 5 min lockout
        remaining_sec = int(300 - (current_time - last_failed_time))
        return False, None, f"Account temporarily locked due to repeated failed login attempts. Try again in {remaining_sec} seconds or reset password."

    if verify_password(password, user_record["password_hash"], user_record["salt"]):
        # Reset failed attempts on success
        user_record["failed_attempts"] = 0
        user_record["last_failed_time"] = 0
        _save_users(users)
        
        user_info = {
            "full_name": user_record["full_name"],
            "email": user_record["email"],
            "role": user_record.get("role", "researcher"),
            "created_at": user_record.get("created_at", "N/A"),
            "login_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return True, user_info, "Authentication successful."
    else:
        # Increment failed attempts
        user_record["failed_attempts"] = failed_attempts + 1
        user_record["last_failed_time"] = current_time
        _save_users(users)
        return False, None, "Invalid email or password."


def reset_password(email: str, new_password: str, confirm_password: str) -> Tuple[bool, str]:
    """Resets user password after email verification and password validation."""
    clean_email = email.strip().lower()

    if not clean_email or not validate_email(clean_email):
        return False, "Please enter a valid registered email address."
    if not new_password:
        return False, "New password is required."
    if new_password != confirm_password:
        return False, "Passwords do not match."

    pw_eval = check_password_strength(new_password)
    if not pw_eval["is_acceptable"]:
        return False, f"New password is too weak: {', '.join(pw_eval['missing'])}"

    users = _load_users()
    if clean_email not in users:
        return False, "No account found with this email address."

    salt = os.urandom(SALT_SIZE).hex()
    password_hash = hash_password(new_password, salt)

    users[clean_email]["password_hash"] = password_hash
    users[clean_email]["salt"] = salt
    users[clean_email]["failed_attempts"] = 0
    users[clean_email]["last_failed_time"] = 0
    users[clean_email]["password_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if _save_users(users):
        return True, "Password reset successful! You can now log in with your new password."
    else:
        return False, "Failed to update password in user database."

