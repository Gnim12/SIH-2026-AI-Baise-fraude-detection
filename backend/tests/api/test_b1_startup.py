"""B1b: startup fails loudly (artefact dir unwritable, model weights absent
or hash-mismatched) rather than lazily on a traveller's first request."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.registry import ModelRegistryError


def test_startup_fails_when_artefact_directory_is_unwritable(tmp_path, monkeypatch):
    from app.config import settings

    # A POSIX chmod-based "read-only directory" doesn't reliably block
    # writes on Windows (the read-only file attribute isn't enforced for
    # directory contents there). Pointing the artefact root through a plain
    # FILE component instead -- b1_artefact_root's own mkdir(parents=True)
    # call cannot succeed under a path component that is a file -- fails
    # for the same underlying reason ("this location cannot be written to")
    # on every platform.
    blocker_file = tmp_path / "not_a_directory"
    blocker_file.write_text("blocker", encoding="utf-8")
    unwritable = blocker_file / "artefacts"

    monkeypatch.setattr(settings, "b1_artefact_root", unwritable)
    with pytest.raises((RuntimeError, OSError)):
        with TestClient(app):
            pass


def test_startup_fails_when_weights_are_absent(monkeypatch):
    import app.main as main_module

    def _raise_missing(*args, **kwargs):
        raise ModelRegistryError("mrz_crnn is listed in the manifest but the file does not exist")

    monkeypatch.setattr(main_module, "verify_all", _raise_missing)
    with pytest.raises(ModelRegistryError):
        with TestClient(app):
            pass


def test_startup_fails_when_a_hash_mismatches(monkeypatch):
    import app.main as main_module

    def _raise_mismatch(*args, **kwargs):
        raise ModelRegistryError("mrz_crnn hash mismatch")

    monkeypatch.setattr(main_module, "verify_all", _raise_mismatch)
    with pytest.raises(ModelRegistryError):
        with TestClient(app):
            pass
