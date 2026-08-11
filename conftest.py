"""Root conftest: put the repo root on sys.path so tests can ``import violations.*`` regardless of
where pytest is invoked from."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def pytest_configure(config):
    """Register the marks used by tests/test_docker_stack.py so a plain run does not
    emit PytestUnknownMarkWarning for them."""
    config.addinivalue_line("markers", "docker: needs a running Docker daemon")
    config.addinivalue_line(
        "markers", "docker_build: builds a container image - slow, needs --rundocker-build")


def pytest_addoption(parser):
    parser.addoption(
        "--rundocker-build", action="store_true", default=False,
        help="run the image-build tests (slow: the CPU worker image pulls ~1 GB of wheels)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--rundocker-build"):
        return
    skip = __import__("pytest").mark.skip(reason="needs --rundocker-build")
    for item in items:
        if "docker_build" in item.keywords:
            item.add_marker(skip)
