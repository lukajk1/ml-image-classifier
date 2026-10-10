import pytest

from conftest import write_json
from runtime_paths import asset_root


def requirement(root, runtime_id):
    write_json(root / "runtime-requirements.json", {"format": 1, "runtimeId": runtime_id})


def test_defaults_to_code_root(tmp_path):
    requirement(tmp_path, "1")
    assert asset_root(tmp_path) == tmp_path


def test_environment_override(tmp_path, monkeypatch):
    requirement(tmp_path, "1")
    pack = tmp_path / "pack"
    pack.mkdir()
    monkeypatch.setenv("UMBRA_ML_ASSETS", str(pack))
    assert asset_root(tmp_path) == pack


def test_matching_runtime_marker_is_accepted(tmp_path, monkeypatch):
    requirement(tmp_path, "7")
    pack = tmp_path / "pack"
    write_json(pack / "ml-runtime.json", {"format": 1, "runtimeId": "7"})
    monkeypatch.setenv("UMBRA_ML_ASSETS", str(pack))
    assert asset_root(tmp_path) == pack


@pytest.mark.parametrize("marker", [{"format": 1, "runtimeId": "6"}, {"format": 2, "runtimeId": "7"}])
def test_mismatched_runtime_marker_is_rejected(tmp_path, monkeypatch, marker):
    requirement(tmp_path, "7")
    pack = tmp_path / "pack"
    write_json(pack / "ml-runtime.json", marker)
    monkeypatch.setenv("UMBRA_ML_ASSETS", str(pack))
    with pytest.raises(ValueError, match="runtime pack 7"):
        asset_root(tmp_path)


def test_unmarked_baseline_only_satisfies_runtime_1(tmp_path):
    requirement(tmp_path, "2")
    with pytest.raises(ValueError, match="runtime pack 2"):
        asset_root(tmp_path)
    (tmp_path / ".venv").mkdir()  # A development checkout is trusted as-is.
    assert asset_root(tmp_path) == tmp_path


def test_no_requirement_file_skips_checks(tmp_path):
    assert asset_root(tmp_path) == tmp_path
