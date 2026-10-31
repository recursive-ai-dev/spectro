"""Guard the declared ``requires-python = ">=3.9"`` support floor.

The package advertises Python 3.9 support but CI only exercised 3.11/3.12, so
3.10+-only syntax or annotations could land unnoticed.  These tests parse every
shipped module with a 3.9 grammar and reject runtime-evaluated PEP 585/604
annotations in modules that do not opt into postponed evaluation.
"""

import ast
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("src/ssgs", "scripts", "examples", "tests")
POSTPONED = "from __future__ import annotations"
GENERIC_BUILTINS = {"list", "dict", "set", "tuple", "frozenset", "type"}


def _modules():
    files = []
    for directory in SOURCE_ROOTS:
        files.extend(sorted((ROOT / directory).rglob("*.py")))
    assert files, "no Python modules found; repository layout changed?"
    return files


def _has_postponed_annotations(tree):
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _runtime_annotations(tree):
    """Yield (lineno, description) for annotations evaluated at import time."""
    for node in ast.walk(tree):
        candidates = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                candidates.append(node.returns)
            candidates.extend(
                argument.annotation
                for argument in ast.walk(node.args)
                if isinstance(argument, ast.arg) and argument.annotation is not None
            )
        elif isinstance(node, ast.AnnAssign):
            candidates.append(node.annotation)
        for annotation in candidates:
            for child in ast.walk(annotation):
                if isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr):
                    yield child.lineno, "PEP 604 union (X | Y)"
                elif (
                    isinstance(child, ast.Subscript)
                    and isinstance(child.value, ast.Name)
                    and child.value.id in GENERIC_BUILTINS
                ):
                    yield child.lineno, f"PEP 585 generic ({child.value.id}[...])"


@pytest.mark.parametrize(
    "path", _modules(), ids=lambda path: str(path.relative_to(ROOT))
)
def test_module_parses_with_the_python_39_grammar(path):
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path), feature_version=(3, 9))
    except SyntaxError as error:  # pragma: no cover - failure path
        pytest.fail(f"{path.relative_to(ROOT)}:{error.lineno}: {error.msg}")

    if _has_postponed_annotations(tree):
        return
    offenders = sorted(set(_runtime_annotations(tree)))
    assert not offenders, (
        f"{path.relative_to(ROOT)} uses Python 3.10+ annotations without "
        f"`{POSTPONED}`: {offenders}"
    )


def test_ci_matrix_covers_the_declared_minimum_versions():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requires = re.search(r'requires-python\s*=\s*"([^"]+)"', pyproject)
    assert requires, "pyproject.toml no longer declares requires-python"
    minimum = tuple(int(part) for part in re.findall(r"\d+", requires.group(1))[:2])

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    matrix = re.search(r"python:\s*\[([^\]]+)\]", workflow)
    assert matrix, "CI workflow no longer defines a Python matrix"
    versions = {
        tuple(int(part) for part in re.findall(r"\d+", entry)[:2])
        for entry in matrix.group(1).split(",")
    }
    assert minimum in versions, (
        f"requires-python declares {minimum} but the CI matrix is {sorted(versions)}"
    )
    assert (minimum[0], minimum[1] + 1) in versions, (
        "the second-oldest supported interpreter should be tested as well"
    )
