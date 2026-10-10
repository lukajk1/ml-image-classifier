"""Locate immutable model assets independently of the updatable backend code."""
import json
import os
from pathlib import Path


def asset_root(code_root):
    code_root = Path(code_root)
    assets = Path(os.environ.get('UMBRA_ML_ASSETS', str(code_root)))
    requirement = code_root / 'runtime-requirements.json'
    if requirement.is_file():
        expected = json.loads(requirement.read_text(encoding='utf-8'))
        marker = assets / 'ml-runtime.json'
        if marker.is_file():
            actual = json.loads(marker.read_text(encoding='utf-8'))
            if actual.get('format') != 1 or actual.get('runtimeId') != expected['runtimeId']:
                raise ValueError(f"Install Umbra Tags ML runtime pack {expected['runtimeId']}.")
        elif expected['runtimeId'] != '1' and not (assets / '.venv').is_dir():
            raise ValueError(f"Install Umbra Tags ML runtime pack {expected['runtimeId']}.")
        # The original full offline installer shipped this baseline without a marker.
    return assets
