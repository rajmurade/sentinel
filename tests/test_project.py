"""Project-level checks that keep the deployment contract intact.

These are the things that silently break a Hugging Face Space: a missing
license, an unignored secret, or a sample dataset that was never committed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
LICENSE_PATH: Path = PROJECT_ROOT / "LICENSE"
SAMPLE_CSV: Path = PROJECT_ROOT / "data" / "sample.csv"
REQUIREMENTS: Path = PROJECT_ROOT / "requirements.txt"
CONTEXT: Path = PROJECT_ROOT / "CONTEXT.md"
README: Path = PROJECT_ROOT / "README.md"

MUST_BE_IGNORED: list[str] = [".env", ".chroma/", "alerts_log.jsonl"]


def is_ignored(path: str) -> bool:
    """Whether git would ignore ``path``, per the repo's .gitignore rules."""
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def test_license_is_mit() -> None:
    """The project ships an MIT license."""
    assert LICENSE_PATH.is_file(), "LICENSE is missing"
    text = LICENSE_PATH.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "WITHOUT WARRANTY OF ANY KIND" in text


@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_secrets_and_runtime_state_are_ignored(path: str) -> None:
    """Secrets, the vector store, and the alert log can never be committed."""
    assert is_ignored(path), f"{path} is not git-ignored"


def test_sample_dataset_is_committed() -> None:
    """Demo mode needs data/sample.csv in the repository, not generated at runtime."""
    assert SAMPLE_CSV.is_file()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "data/sample.csv"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, "data/sample.csv is not tracked by git"


def test_requirements_pin_python_312_without_torch() -> None:
    """Deployable pins, and no PyTorch anywhere in the dependency tree."""
    text = REQUIREMENTS.read_text(encoding="utf-8")
    # Only non-comment, non-blank lines count as dependencies.
    packages = [
        line.split("#", 1)[0].strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    names = [package.split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].strip() for package in packages if package]
    assert "sentence-transformers" not in names
    assert "torch" not in names
    for package in ("streamlit", "pandas", "numpy", "chromadb", "onnxruntime", "requests"):
        assert package in names


def test_readme_has_hugging_face_spaces_header() -> None:
    """The Spaces YAML block has to be the very first thing in the file."""
    text = README.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    header = text.split("---", 2)[1]
    for key in ("title: Sentinel", "sdk: streamlit", "app_file: app.py", 'python_version: "3.12"'):
        assert key in header


def test_context_describes_the_onnx_embedding() -> None:
    """CONTEXT.md must not still claim sentence-transformers."""
    text = CONTEXT.read_text(encoding="utf-8")
    memory_row = next(line for line in text.splitlines() if line.startswith("| `core/memory.py`"))
    assert "ONNX" in memory_row
    assert "TF-IDF" in memory_row
