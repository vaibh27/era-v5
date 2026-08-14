from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arithmetic.reference.limbs import int_to_limbs, limbs_to_int, normalize_limbs


def test_limb_round_trip_and_documented_example():
    assert int_to_limbs(123456789, 1000) == [789, 456, 123]
    for value in (0, 1, 999, 1000, 123456789, 10**30):
        assert limbs_to_int(int_to_limbs(value, 1000), 1000) == value


def test_normalization_removes_padded_high_zeros():
    assert normalize_limbs([789, 456, 123, 0, 0], 1000) == [789, 456, 123]
