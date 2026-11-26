"""Checks of the package files that a PyPI upload sends (sdist and wheel)."""

import email
import importlib.util
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SDIST_TOP_LEVEL = {"LICENSE", "README.md", "PKG-INFO", "pyproject.toml", "setup.cfg", "flowmop", "flowmop.egg-info"}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build the sdist and the wheel once, as `python -m build` does for an upload."""
    if importlib.util.find_spec("build") is None:
        pytest.skip("the 'build' package is not installed")
    source = tmp_path_factory.mktemp("source") / "repo"
    shutil.copytree(REPO, source, ignore=shutil.ignore_patterns(".git", ".worktrees", "*.egg-info", "build", "dist"))
    # A hidden folder with a private file: the sdist must not take it.
    (source / ".private_tool").mkdir(exist_ok=True)
    (source / ".private_tool" / "private.txt").write_text("private\n")
    out = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out), str(source)],
        capture_output=True,
        text=True,
    )
    log = result.stdout + result.stderr
    if result.returncode != 0 and "Could not find a version" in log:
        pytest.skip("no network for the isolated build environment")
    assert result.returncode == 0, log
    wheel = next(out.glob("*.whl"))
    sdist = next(out.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(archive.read(metadata_name))
        wheel_license = next(
            archive.read(n).decode() for n in archive.namelist() if n.endswith(".dist-info/licenses/LICENSE")
        )
    with tarfile.open(sdist) as archive:
        sdist_names = [name.split("/", 1)[1] for name in archive.getnames() if "/" in name]
    return {"log": log, "metadata": metadata, "wheel_license": wheel_license, "sdist_names": sdist_names}


def test_license_file_is_mit():
    text = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2025-2026 Tony Xu" in text
    assert "GNU GENERAL PUBLIC LICENSE" not in text


def test_wheel_metadata_has_the_pypi_fields(built):
    metadata = built["metadata"]
    assert metadata["Name"] == "flowmop"
    assert metadata["License-Expression"] == "MIT"
    assert metadata["Description-Content-Type"] == "text/markdown"
    classifiers = metadata.get_all("Classifier") or []
    for classifier in (
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ):
        assert classifier in classifiers
    assert not any(c.startswith("License ::") for c in classifiers)
    assert "flow cytometry" in metadata["Keywords"]
    urls = metadata.get_all("Project-URL") or []
    assert "Issues, https://github.com/1ordinateur/FlowMOP/issues" in urls
    assert built["wheel_license"].startswith("MIT License")


def test_sdist_has_only_package_files(built):
    names = built["sdist_names"]
    for needed in ("flowmop/cli.py", "LICENSE", "README.md", "pyproject.toml"):
        assert needed in names
    hidden = [n for n in names if any(part.startswith(".") for part in Path(n).parts)]
    assert hidden == []
    other = sorted({Path(n).parts[0] for n in names if n} - SDIST_TOP_LEVEL)
    assert other == []  # no tests/: they need the wrapper scripts, which the sdist does not have


def test_test_extra_installs_the_build_tool():
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    test_extra = pyproject.split("test = [", 1)[1].split("]", 1)[0]
    assert '"build"' in test_extra


def test_readme_installs_from_pypi_and_names_the_license():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    install = readme.split("## Install", 1)[1].split("\n## ", 1)[0]
    assert "pip install flowmop" in install
    assert install.index("pip install flowmop") < install.index("pip install git+")
    assert "## License" in readme
    license_section = readme.split("## License", 1)[1]
    assert "MIT" in license_section
