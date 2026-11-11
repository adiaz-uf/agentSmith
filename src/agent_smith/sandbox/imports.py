"""Guarded import hook enforcing allowlist-based module access for the sandbox."""

from __future__ import annotations

import builtins
import types
from collections.abc import Callable
from typing import Any


def is_import_allowed(name: str, authorized_imports: list[str]) -> bool:
    """Check if a module name matches any entry in the authorized imports list.

    Rules:
    - Exact match: 'json' allows 'json'.
    - Wildcard: 'collections.*' allows 'collections.abc' and deeper,
      but does NOT allow 'collections' itself.
    """
    for pattern in authorized_imports:
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            if name.startswith(f"{prefix}."):
                return True
        elif name == pattern:
            return True
    return False


def create_safe_import(
    authorized_imports: list[str],
    real_import: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Create a guarded __import__ function enforcing the authorized_imports allowlist."""
    underlying_import = real_import or builtins.__import__
    allowed_summary = ", ".join(authorized_imports)

    def safe_import(
        name: str,
        globals: dict[str, Any] | None = None,  # noqa: A002
        locals: dict[str, Any] | None = None,  # noqa: A002
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        # 1. Reject relative imports
        if level > 0:
            raise ImportError(
                f"Relative imports are not allowed (level={level}). "
                f"Authorized imports: {allowed_summary}"
            )

        # 2. Check top-level / target module name
        if not is_import_allowed(name, authorized_imports):
            raise ImportError(
                f"import of '{name}' is not allowed. "
                f"Authorized imports: {allowed_summary}"
            )

        # 3. Delegate to the original Python import
        mod = underlying_import(name, globals, locals, fromlist, level)

        # 4. Check submodules or leaked modules in `from X import Y`
        if fromlist:
            for item in fromlist:
                if item == "*":
                    continue
                # If the imported attribute is itself a module, check it
                attr = getattr(mod, item, None)
                if isinstance(attr, types.ModuleType):
                    sub_name = getattr(attr, "__name__", f"{name}.{item}")
                    if not is_import_allowed(sub_name, authorized_imports):
                        raise ImportError(
                            f"import of '{sub_name}' is not allowed. "
                            f"Authorized imports: {allowed_summary}"
                        )

        return mod

    return safe_import

