import pytest
from lingshu.core.evidence import _check_weight

def test_check_weight_huge_int_overflow():
    """dsh-bounty #2219: Huge integers that overflow C float conversion must raise ValueError instead of unhandled OverflowError."""
    huge_int = 10**400
    with pytest.raises(ValueError) as excinfo:
        _check_weight(huge_int)
    assert "evidence weight 必须是数值" in str(excinfo.value)

def test_check_weight_valid_numbers():
    """dsh-bounty #2219: Normal valid weights in (0, 1] work properly."""
    assert _check_weight(1.0) == 1.0
    assert _check_weight(0.5) == 0.5
    assert _check_weight("0.8") == 0.8
    assert _check_weight(1) == 1.0

def test_check_weight_invalid_ranges():
    """dsh-bounty #2219: Weights <= 0 or > 1 or non-finite raise ValueError with range error message."""
    for bad in [0.0, -0.5, 1.1, float('nan'), float('inf')]:
        with pytest.raises(ValueError) as excinfo:
            _check_weight(bad)
        assert "evidence weight 须在 (0, 1] 且有限" in str(excinfo.value)
