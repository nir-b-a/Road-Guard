"""Root conftest: put the repo root on sys.path so tests can ``import violations.*`` regardless of
where pytest is invoked from."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
