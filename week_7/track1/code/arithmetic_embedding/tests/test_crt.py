from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from crt import CRTSystem, verify_exhaustive


def test_crt_basic():
    system = CRTSystem((5, 7, 11))
    assert system.capacity == 385
    for x in range(system.capacity):
        assert system.decode(system.encode(x)) == x


def test_operations():
    system = CRTSystem((5, 7, 11))
    for x in range(system.capacity):
        for y in range(system.capacity):
            assert system.add(system.encode(x), system.encode(y)) == system.encode((x + y) % system.capacity)
            assert system.mul(system.encode(x), system.encode(y)) == system.encode((x * y) % system.capacity)


def test_exhaustive_helper():
    verify_exhaustive(CRTSystem((3, 5, 7)))


def test_operations_reject_wrong_residue_count():
    system = CRTSystem((5, 7, 11))
    try:
        system.add((1, 2), (3, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("add should reject incomplete residue vectors")

    try:
        system.mul((1, 2, 3), (4, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("mul should reject incomplete residue vectors")
