"""P4-G0 Claim-to-Artifact Evidence Audit package.

This package implements the P4-G0 phase: a forensic audit of all headline
claims in manuscript v3 against the actual code, data, and result artifacts.

The audit does NOT train any new model and does NOT create new performance
claims.  It only verifies, recomputes, and flags inconsistencies.
"""

from .run_claim_audit import (
    Claim,
    run_claim_audit,
    build_claim_registry,
    verify_claims,
    write_outputs,
    ALLOWED_STATUSES,
)

__all__ = [
    "Claim",
    "run_claim_audit",
    "build_claim_registry",
    "verify_claims",
    "write_outputs",
    "ALLOWED_STATUSES",
]
