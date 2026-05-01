import pytest

from chinese_remainder_theorem import ChineseRemainderTheorem


@pytest.mark.parametrize(
    "n1,r1,n2,r2,expected",
    [
        (686579304, 295310485, 26855093, 8217207, 579510303168901),
        (11, 3, 17, 7, 58),
    ],
)
def test_chinese_remainder_theorem(n1, r1, n2, r2, expected):
    assert expected == ChineseRemainderTheorem(n1, r1, n2, r2)
