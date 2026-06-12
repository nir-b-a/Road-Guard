"""
NumPy compatibility shim for UnLanedet.

UnLanedet was written against NumPy < 1.24 and still uses the deprecated scalar
aliases (`np.bool`, `np.int`, `np.float`, `np.long`, ...) that NumPy 1.24
removed. Our env pins numpy 1.26.4 (required by the rest of the stack), so those
attribute accesses raise `AttributeError` deep inside UnLanedet's lane decoding
(e.g. `clr_head.predictions_to_pred`).

We can't pin numpy down far enough to bring the aliases back without breaking
everything else, and patching the cloned framework is fragile (the clone is
gitignored and re-cloning would lose the edit). So we restore the aliases on the
`numpy` module at runtime. Each maps to the plain Python builtin, exactly as the
old NumPy aliases did, so behaviour is unchanged.

Call `install()` once before exercising any UnLanedet code path.
"""
from __future__ import annotations

import numpy as np

# Deprecated alias -> replacement (matches pre-1.24 NumPy semantics).
_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "long": int,   # np.long was a Python-int alias on Py3
    "unicode": str,
}


def install() -> None:
    """Restore the removed NumPy scalar aliases if they're missing (idempotent)."""
    for name, builtin in _ALIASES.items():
        if not hasattr(np, name):
            setattr(np, name, builtin)
