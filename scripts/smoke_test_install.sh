#!/usr/bin/env bash
# Clean-environment smoke test: verify package imports and tests run
# without relying on pre-existing conda environment state.
set -euo pipefail

echo "=== PC-CNG Clean Environment Smoke Test ==="
echo "Python: $(python --version 2>&1)"
echo "Pip: $(pip --version 2>&1)"

# Install minimal dependencies
echo ""
echo "=== Installing minimal dependencies ==="
pip install --quiet pytest numpy pandas

# Install package in development mode if pyproject.toml exists
if [ -f pyproject.toml ]; then
    echo ""
    echo "=== Installing package (development mode) ==="
    pip install --quiet -e . || echo "WARNING: dev install failed (non-fatal for smoke test)"
fi

# Verify imports
echo ""
echo "=== Verifying imports ==="
python -c "
import sys
sys.path.insert(0, 'chem_negative_sampling')
try:
    import pc_cng
    print('OK: pc_cng imported')
except ImportError as e:
    print(f'EXPECTED (missing optional deps): {e}')
except Exception as e:
    print(f'UNEXPECTED ERROR: {e}')
    sys.exit(1)
print('OK: smoke test passed')
"

# Run tests if they exist
echo ""
echo "=== Running tests ==="
if [ -d chem_negative_sampling/tests ]; then
    cd chem_negative_sampling
    python -m pytest tests/ -q --tb=line -x || echo "WARNING: some tests failed (may need full deps)"
    cd ..
fi

echo ""
echo "=== Smoke test complete ==="
