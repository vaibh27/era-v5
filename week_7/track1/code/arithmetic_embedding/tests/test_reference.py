from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arithmetic.reference.bigint import add, divmod_exact, mul, sub


def test_bigint_oracle_matches_python_integer_arithmetic():
    assert add(123456789, 987654321) == 1111111110
    assert sub(10, 13) == -3
    assert mul(49898, 233234) == 11637910132
    assert divmod_exact(100, 9) == (11, 1)
    with pytest.raises(ZeroDivisionError):
        divmod_exact(1, 0)
