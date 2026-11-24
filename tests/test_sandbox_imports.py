"""Unit tests for import allowlist enforcement in the sandbox (Issue #19)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_smith.core.models import SandboxConfig
from agent_smith.sandbox.imports import is_import_allowed
from agent_smith.sandbox.runner import ProcessSandbox


def test_is_import_allowed_matching_logic() -> None:
    """Test exact and wildcard matching rules."""
    allowed = ["math", "collections.*", "json"]

    # Exact matches
    assert is_import_allowed("math", allowed) is True
    assert is_import_allowed("json", allowed) is True

    # Wildcard matches submodules and descendants
    assert is_import_allowed("collections.abc", allowed) is True
    assert is_import_allowed("collections.abc.sub", allowed) is True

    # Wildcard does NOT match the root package if not explicitly listed
    assert is_import_allowed("collections", allowed) is False

    # Blocked modules
    assert is_import_allowed("os", allowed) is False
    assert is_import_allowed("sys", allowed) is False
    assert is_import_allowed("subprocess", allowed) is False


def test_default_modules_import_successfully() -> None:
    """Verify that default allowed modules can be imported and executed."""
    code = (
        "import math\n"
        "import collections\n"
        "import collections.abc\n"
        "import itertools\n"
        "import re\n"
        "import json\n"
        "import typing\n"
        "import functools\n"
        "import operator\n"
        "import heapq\n"
        "import bisect\n"
        "import copy\n"
        "import string\n"
        "import random\n"
        "import datetime\n"
        "import array\n"
        "import cmath\n"
        "success = True"
    )
    with ProcessSandbox() as sandbox:
        res = sandbox.execute(code)
        assert res.success is True
        assert res.error is None


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import sys",
        "import subprocess",
        "import socket",
        "import ctypes",
        "import importlib",
        "from os import path",
        "__import__('os')",
        "__import__('sys')",
    ],
)
def test_unauthorized_imports_raise_import_error(code: str) -> None:
    """Verify that blocked modules fail with a clear, readable ImportError."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute(code)
        assert res.success is False
        assert res.error_type == "ImportError"
        assert "not allowed" in (res.error or "").lower()
        assert "authorized imports:" in (res.error or "").lower()


def test_relative_imports_are_rejected() -> None:
    """Verify that relative imports with level > 0 are rejected."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute("__import__('math', level=1)")
        assert res.success is False
        assert res.error_type == "ImportError"
        assert "relative imports are not allowed" in (res.error or "").lower()


def test_module_attribute_leaks_are_blocked() -> None:
    """Verify that importing leaked dangerous modules via 'from ... import ...' fails."""
    with ProcessSandbox() as sandbox:
        # random leaks os as '_os'
        res1 = sandbox.execute("from random import _os")
        assert res1.success is False
        assert res1.error_type == "ImportError"
        assert "import of 'os' is not allowed" in (res1.error or "")

        # datetime leaks sys as 'sys'
        res2 = sandbox.execute("from datetime import sys")
        assert res2.success is False
        assert res2.error_type == "ImportError"
        assert "import of 'sys' is not allowed" in (res2.error or "")


def test_aliased_and_star_imports() -> None:
    """Verify that aliased imports and star imports work for allowed modules."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute(
            "import math as m\n"
            "from itertools import *\n"
            "assert m.isclose(m.pi, 3.14159, abs_tol=1e-3)\n"
            "x = list(islice(count(10), 3))\n"
            "assert x == [10, 11, 12]"
        )
        assert res.success is True
        assert res.error is None


def test_wildcard_does_not_imply_root_module() -> None:
    """Verify that 'pkg.*' permits submodules but NOT 'pkg' unless explicitly listed."""
    config = SandboxConfig(authorized_imports=["collections.*"])
    with ProcessSandbox(config=config) as sandbox:
        # Root package should fail
        res_root = sandbox.execute("import collections")
        assert res_root.success is False
        assert res_root.error_type == "ImportError"

        # Submodule should succeed
        res_sub = sandbox.execute(
            "import collections.abc\nassert hasattr(collections.abc, 'Iterable')"
        )
        assert res_sub.success is True


def test_import_state_persists_across_executions() -> None:
    """Verify that an allowed import remains available in subsequent executions,

    and a rejected import does not crash the persistent namespace.
    """
    with ProcessSandbox() as sandbox:
        # Step 1: Import allowed module
        res1 = sandbox.execute("import math\ndef circle_area(r): return math.pi * r * r")
        assert res1.success is True

        # Step 2: Attempt illegal import (fails)
        res2 = sandbox.execute("import os")
        assert res2.success is False
        assert res2.error_type == "ImportError"

        # Step 3: Previous definitions are intact and worker is alive
        res3 = sandbox.execute("print(circle_area(2))")
        assert res3.success is True
        assert "12.56" in res3.stdout


def test_dynamic_reconfiguration_via_json(tmp_path: Path) -> None:
    """Verify that modifying authorized_imports changes behavior dynamically."""
    custom_config = tmp_path / "sandbox_custom.json"
    custom_config.write_text(
        '{"authorized_imports": ["operator", "operator.*"], "max_execution_time_seconds": 15}',
        encoding="utf-8",
    )

    config = SandboxConfig.from_file(custom_config)
    with ProcessSandbox(config=config) as sandbox:
        # operator is allowed
        res1 = sandbox.execute("import operator\nres = operator.mul(6, 7)")
        assert res1.success is True

        # math is NOT in this custom configuration
        res2 = sandbox.execute("import math")
        assert res2.success is False
        assert res2.error_type == "ImportError"


@pytest.mark.parametrize(
    "code",
    [
        "import random\nrandom._os",
        "import typing\ntyping.sys",
        "import datetime\ndatetime.sys",
        "import collections\ncollections._sys",
        "import json\njson.decoder",
        "import random\ngetattr(random, '_' + 'os')",
    ],
)
def test_module_attribute_access_leaks_are_blocked(code: str) -> None:
    """Plain attribute access on an allowed module must not reach non-allowed modules."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute(code)
        assert res.success is False
        assert res.error_type == "AttributeError"
        assert "not allowed" in (res.error or "")


def test_string_re_leak_is_blocked() -> None:
    """string._re is the (allowed) re module, but re's own internals stay guarded."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute("import string\nstring._re.sre_compile")
        assert res.success is False
        assert res.error_type == "AttributeError"


@pytest.mark.parametrize(
    "attr",
    ["__builtins__", "__loader__", "__spec__", "__dict__", "__file__"],
)
def test_module_dunder_access_is_blocked(attr: str) -> None:
    """Module dunders leading back to the real import machinery are not reachable."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute(f"import random\nv = random.{attr}\nv['__import__']('os')")
        assert res.success is False
        assert "os" not in (res.return_value or "")
        assert res.error_type in {"AttributeError", "TypeError", "KeyError"}


def test_allowed_submodule_attribute_access_still_works() -> None:
    """Reaching an authorized module through an attribute keeps working."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute(
            "import collections, typing\n"
            "assert collections.abc.Iterable is not None\n"
            "typing.collections.Counter('aab')"
        )
        assert res.success is True, res.error
        assert res.return_value == "Counter({'a': 2, 'b': 1})"


def test_star_import_skips_blocked_modules() -> None:
    """`from X import *` works for allowed modules without importing leaked ones."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute("from math import *\nfrom datetime import *\nsqrt(16)")
        assert res.success is True, res.error
        assert res.return_value == "4.0"


def test_dotted_import_does_not_grant_root_package() -> None:
    """With only 'pkg.*' allowed, `import pkg.sub` binds pkg but exposes only pkg.sub."""
    config = SandboxConfig(authorized_imports=["collections.*"])
    with ProcessSandbox(config=config) as sandbox:
        ok = sandbox.execute("import collections.abc\ncollections.abc.Iterable.__name__")
        assert ok.success is True, ok.error
        assert ok.return_value == "'Iterable'"

        for code in ("collections.Counter", "collections._sys", "collections.deque"):
            res = sandbox.execute(code)
            assert res.success is False, code
            assert res.error_type == "AttributeError"
            assert "not allowed" in (res.error or "")


@pytest.mark.parametrize("name", ["__loader__", "__spec__"])
def test_builtin_loader_is_not_exposed(name: str) -> None:
    """The BuiltinImporter must not be reachable to load modules like posix."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute(f"{name}")
        assert res.success is False
        assert res.error_type == "NameError"

        res = sandbox.execute(f"__builtins__['{name}']")
        assert res.success is False
        assert res.error_type == "KeyError"


def test_builtin_loader_not_reachable_through_builtin_modules() -> None:
    """itertools.__loader__ is BuiltinImporter in CPython; it must stay hidden."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute("import itertools\nitertools.__loader__.load_module('posix')")
        assert res.success is False
        assert res.error_type == "AttributeError"
