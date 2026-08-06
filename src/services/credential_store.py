"""Encrypted local storage for broker login credentials ("remember me").

Design notes:
- Credentials are encrypted at rest with Fernet (AES-128-CBC + HMAC) using
  a key that lives in a separate, gitignored file from the encrypted data.
  Splitting key and data means a leaked data file alone is useless.
- This protects against casual disk/backup snooping — it is NOT a defense
  against someone with full access to the machine while it's unlocked and
  running (they could read the key file too). For that level of security,
  swap the backend for an OS keyring later; the public interface below
  (save_credentials / load_credentials / clear_credentials) would not need
  to change.
- Nothing here is ever committed to git — both files are covered by
  .gitignore patterns (.credential_key, .credentials.enc).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

from src.logger import get_logger

log = get_logger(__name__)

_KEY_FILE = Path(".credential_key")
_DATA_FILE = Path(".credentials.enc")


def _get_or_create_key() -> bytes:
    from cryptography.fernet import Fernet

    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    _KEY_FILE.write_bytes(key)
    try:
        _KEY_FILE.chmod(0o600)
    except Exception:
        pass  # best-effort on platforms without POSIX permissions (Windows)
    return key


def _load_all() -> Dict[str, Any]:
    if not _DATA_FILE.exists():
        return {}
    try:
        from cryptography.fernet import Fernet
        key = _get_or_create_key()
        fernet = Fernet(key)
        decrypted = fernet.decrypt(_DATA_FILE.read_bytes())
        return json.loads(decrypted.decode("utf-8"))
    except Exception as exc:
        log.warning("Could not read saved credentials, ignoring: %s", exc)
        return {}


def _save_all(data: Dict[str, Any]) -> None:
    from cryptography.fernet import Fernet
    key = _get_or_create_key()
    fernet = Fernet(key)
    encrypted = fernet.encrypt(json.dumps(data).encode("utf-8"))
    _DATA_FILE.write_bytes(encrypted)
    try:
        _DATA_FILE.chmod(0o600)
    except Exception:
        pass


def save_credentials(broker: str, login: str, password: str, server: str) -> bool:
    """Save (encrypted) credentials for a broker. Returns True on success."""
    try:
        data = _load_all()
        data[broker] = {"login": login, "password": password, "server": server}
        _save_all(data)
        return True
    except Exception as exc:
        log.error("Failed to save credentials: %s", exc)
        return False


def load_credentials(broker: str) -> Optional[Dict[str, str]]:
    """Load saved credentials for a broker, or None if not saved / unreadable."""
    data = _load_all()
    return data.get(broker)


def clear_credentials(broker: Optional[str] = None) -> None:
    """Clear saved credentials for one broker, or all brokers if none given."""
    if broker is None:
        _DATA_FILE.unlink(missing_ok=True)
        return
    data = _load_all()
    data.pop(broker, None)
    _save_all(data)
