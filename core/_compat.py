"""Compatibility shims for third-party libraries whose API has drifted from
the pinned scikit-learn version.

factor_analyzer 0.5.1 (latest release) calls
`check_array(X, force_all_finite=...)`. scikit-learn renamed that parameter
to `ensure_all_finite` in 1.6 and removed the old name later; since no
factor_analyzer release targets the newer scikit-learn, and no scikit-learn
version old enough to still have `force_all_finite` ships a Python 3.14
wheel, we patch factor_analyzer's bound reference instead of pinning either
package.
"""

from __future__ import annotations

import sklearn.utils as _sklearn_utils

_original_check_array = _sklearn_utils.check_array
_patched = False


def _check_array_compat(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _original_check_array(*args, **kwargs)


def patch_factor_analyzer() -> None:
    global _patched
    if _patched:
        return
    import factor_analyzer.factor_analyzer as fa_module

    fa_module.check_array = _check_array_compat
    _patched = True
