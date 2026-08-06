"""Tests for credential_store.py — encrypted local 'remember me' storage."""

import pytest
from pathlib import Path

from src.services import credential_store as cs


@pytest.fixture(autouse=True)
def _isolated_files(tmp_path, monkeypatch):
    """Point the store at a temp directory so tests never touch real files."""
    monkeypatch.setattr(cs, "_KEY_FILE", tmp_path / ".credential_key")
    monkeypatch.setattr(cs, "_DATA_FILE", tmp_path / ".credentials.enc")
    yield


class TestCredentialStore:
    def test_save_and_load_roundtrip(self):
        ok = cs.save_credentials("mt5", "12345", "s3cret", "VantageInternational-Live")
        assert ok is True
        loaded = cs.load_credentials("mt5")
        assert loaded == {"login": "12345", "password": "s3cret", "server": "VantageInternational-Live"}

    def test_load_missing_returns_none(self):
        assert cs.load_credentials("mt5") is None

    def test_data_file_is_actually_encrypted(self):
        cs.save_credentials("mt5", "12345", "s3cret_password", "SomeServer")
        raw = cs._DATA_FILE.read_bytes()
        assert b"s3cret_password" not in raw

    def test_multiple_brokers_independent(self):
        cs.save_credentials("mt5", "111", "pw1", "ServerA")
        cs.save_credentials("vantage", "222", "pw2", "ServerB")
        assert cs.load_credentials("mt5")["login"] == "111"
        assert cs.load_credentials("vantage")["login"] == "222"

    def test_clear_single_broker(self):
        cs.save_credentials("mt5", "111", "pw1", "ServerA")
        cs.save_credentials("vantage", "222", "pw2", "ServerB")
        cs.clear_credentials("mt5")
        assert cs.load_credentials("mt5") is None
        assert cs.load_credentials("vantage") is not None

    def test_clear_all(self):
        cs.save_credentials("mt5", "111", "pw1", "ServerA")
        cs.clear_credentials()
        assert cs.load_credentials("mt5") is None

    def test_corrupted_file_fails_soft(self):
        cs.save_credentials("mt5", "111", "pw1", "ServerA")
        cs._DATA_FILE.write_bytes(b"not valid encrypted data")
        assert cs.load_credentials("mt5") is None
