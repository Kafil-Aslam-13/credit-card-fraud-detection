"""Generic helper functions used across multiple layers."""

import json
import os

import joblib


def save_joblib(obj, path: str) -> None:
    """Persist any Python object (model, scaler, etc.) to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: str):
    """Load a previously persisted object from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No artifact found at: {path}")
    return joblib.load(path)


def save_json(data: dict, path: str) -> None:
    """Persist a dict as a formatted JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=str)


def load_json(path: str) -> dict:
    """Load a dict from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)