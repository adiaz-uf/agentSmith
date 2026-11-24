"""Guarded import hook enforcing allowlist-based module access for the sandbox."""

from __future__ import annotations

import builtins
import types
from collections.abc import Callable
from typing import Any

# Module dunders that lead back to unguarded import machinery: the real
# builtins (with the real ``__import__``) and the module's loader/spec. The
# proxy's own ``__dict__``/``__loader__``/``__spec__`` never expose the real ones.
_BLOCKED_DUNDERS = frozenset({"__builtins__", "__loader__", "__spec__"})


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


def has_authorized_descendant(name: str, authorized_imports: list[str]) -> bool:
    """Return True if some authorized entry lives strictly below ``name``.

    ``collections.*`` and ``collections.abc`` both make ``collections`` a parent
    that must be traversable (``import collections.abc`` binds ``collections``).
    """
    return any(
        pattern.removesuffix(".*") == name or pattern.startswith(f"{name}.")
        for pattern in authorized_imports
    )


def create_safe_import(
    authorized_imports: list[str],
    real_import: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Create a guarded __import__ function enforcing the authorized_imports allowlist.

    The hook never hands the real module objects to sandboxed code. It returns
    guarded proxies instead, so attribute access such as ``random._os`` or
    ``typing.sys`` is checked against the allowlist just like an import.
    """
    underlying_import = real_import or builtins.__import__
    allowed_summary = ", ".join(authorized_imports)
    proxies: dict[tuple[str, bool], types.ModuleType] = {}

    def denied(name: str) -> str:
        return f"import of '{name}' is not allowed. Authorized imports: {allowed_summary}"

    def guard_module(module: types.ModuleType, full_access: bool) -> types.ModuleType:
        """Wrap ``module`` in a proxy that checks every attribute access.

        ``full_access=False`` is used for parent packages that are not authorized
        themselves but have authorized submodules (``pkg.*``): only those
        submodules are reachable through the proxy.
        """
        key = (module.__name__, full_access)
        cached = proxies.get(key)
        if cached is not None:
            return cached

        mod_name = module.__name__

        def resolve(attr: str) -> Any:
            if attr in _BLOCKED_DUNDERS:
                raise AttributeError(f"access to '{mod_name}.{attr}' is not allowed in the sandbox")
            value = getattr(module, attr)
            if isinstance(value, types.ModuleType):
                target = value.__name__
                if is_import_allowed(target, authorized_imports):
                    return guard_module(value, full_access=True)
                if target == f"{mod_name}.{attr}" and has_authorized_descendant(
                    target, authorized_imports
                ):
                    return guard_module(value, full_access=False)
                raise AttributeError(
                    f"access to module '{target}' via '{mod_name}.{attr}' is not allowed. "
                    f"Authorized imports: {allowed_summary}"
                )
            if not full_access:
                raise AttributeError(
                    f"{denied(mod_name)} (only its authorized submodules are accessible)"
                )
            return value

        def public_names() -> list[str]:
            names = getattr(module, "__all__", None)
            if names is None:
                names = [n for n in dir(module) if not n.startswith("_")]
            accessible = []
            for n in names:
                try:
                    resolve(n)
                except AttributeError:
                    continue
                accessible.append(n)
            return accessible

        class GuardedModule(types.ModuleType):
            def __getattr__(self, attr: str) -> Any:
                if attr == "__all__":
                    return public_names()
                return resolve(attr)

            def __dir__(self) -> list[str]:
                return sorted(public_names())

        GuardedModule.__name__ = GuardedModule.__qualname__ = "module"
        proxy = GuardedModule(mod_name, module.__doc__ if full_access else None)
        proxies[key] = proxy
        return proxy

    def safe_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] | list[str] | None = (),
        level: int = 0,
    ) -> Any:
        # 1. Reject relative imports
        if level > 0:
            raise ImportError(
                f"Relative imports are not allowed (level={level}). "
                f"Authorized imports: {allowed_summary}"
            )

        # 2. Check the requested module name
        if not is_import_allowed(name, authorized_imports):
            raise ImportError(denied(name))

        # 3. Delegate to the original Python import
        mod = underlying_import(name, globals, locals, fromlist, level)

        if not fromlist:
            # `import a.b.c` binds the top-level package `a`. Only give full access
            # to it if `a` itself is authorized; otherwise expose just the path
            # down to the authorized submodules.
            top = name.partition(".")[0]
            return guard_module(mod, full_access=is_import_allowed(top, authorized_imports))

        # 4. `from X import Y`: Y may be a submodule or a module leaked as attribute
        for item in fromlist:
            if item == "*":
                continue
            attr = getattr(mod, item, None)
            if isinstance(attr, types.ModuleType) and not is_import_allowed(
                attr.__name__, authorized_imports
            ):
                raise ImportError(denied(attr.__name__))

        return guard_module(mod, full_access=True)

    return safe_import
