'''Tests for the first exercise set.'''
import pytest
import ml_utils as utils
import time


@pytest.mark.parametrize('m', [3, 4, 9])
def test_basic_matvec(m):
    time.sleep(m)
    assert ('Hello World' in utils.foo())


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
