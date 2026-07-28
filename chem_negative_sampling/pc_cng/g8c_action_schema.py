"""Shared action taxonomy for formal G8-C training and evaluation."""
from __future__ import annotations

from enum import IntEnum


class EditType(IntEnum):
    """Structured action classes with explicit formal-only states."""

    ATOM_TRANSMUTATION = 0
    BOND_ORDER_CHANGE = 1
    FORMED_BOND_MIGRATE = 2
    NO_EDIT = 3
    NOT_APPLICABLE = 4
    BOND_BREAK = 5
    BOND_FORM = 6


NUM_EDIT_TYPES = len(EditType)
FORMAL_REAL_EDIT_TYPES = frozenset(
    {
        EditType.BOND_ORDER_CHANGE,
        EditType.BOND_BREAK,
        EditType.BOND_FORM,
        EditType.NO_EDIT,
    }
)
GENERATIVE_EDIT_TYPES = frozenset(
    {
        EditType.ATOM_TRANSMUTATION,
        EditType.BOND_ORDER_CHANGE,
        EditType.FORMED_BOND_MIGRATE,
        EditType.BOND_BREAK,
    }
)


__all__ = [
    "EditType",
    "NUM_EDIT_TYPES",
    "FORMAL_REAL_EDIT_TYPES",
    "GENERATIVE_EDIT_TYPES",
]
