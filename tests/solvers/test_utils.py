import pytest
from src.solver.qaoa.utils import _invert_integer_binary_representation, int_to_bits, bits_to_int
import numpy as np

def test_bits_to_int():
    assert bits_to_int([1, 0, 1]) == 5
    assert bits_to_int([0, 1, 1], reverse=True) == 6
    assert bits_to_int([0, 0, 1, 1]) == 3
    assert bits_to_int([1, 1, 0, 0], reverse=True) == 3

# Test int_to_bits function
def test_int_to_bits():
    np.testing.assert_array_equal(int_to_bits(5, length=3), np.array([1, 0, 1]))
    np.testing.assert_array_equal(int_to_bits(3, length=4), np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(int_to_bits(3, length=4, reverse=True), np.array([1, 1, 0, 0]))
    np.testing.assert_array_equal(int_to_bits(5, length=3, reverse=True), np.array([1, 0, 1])[::-1])

# Test undo of bits_to_int and int_to_bits functions
def test_undo():
    tries = 10
    for _ in range(tries):
        random_int = np.random.randint(low=1, high=1000)
        recovered = bits_to_int(int_to_bits(random_int))
        assert random_int == recovered

    for _ in range(tries):
        random_bits = [np.random.randint(low=0, high=2) for _ in range(tries)]
        recovered = int_to_bits(bits_to_int(random_bits), length=tries)
        np.testing.assert_array_equal(np.array(random_bits), recovered)

# Test _invert_integer_binary_representation function
def test_invert_bits_default():
    assert _invert_integer_binary_representation(6) == 3  # 6 in binary is '110', inverted is '011' which is 3
    assert _invert_integer_binary_representation(5) == 5  # 5 in binary is '101', inverted is '101' which is 5
    assert _invert_integer_binary_representation(10) == 5  # 10 in binary is '1010', inverted is '0101' which is 5
    assert _invert_integer_binary_representation(0) == 0  # 0 in binary is '0', inverted is '0' which is 0

def test_invert_bits_with_num_bits():
    assert _invert_integer_binary_representation(6, 4) == 6  # 6 in 4 bits is '0110', inverted is '0110' which is 6
    assert _invert_integer_binary_representation(5, 4) == 10  # 5 in 4 bits is '0101', inverted is '1010' which is 10
    assert _invert_integer_binary_representation(15, 4) == 15  # 15 in 4 bits is '1111', inverted is '1111' which is 15
    assert _invert_integer_binary_representation(3, 4) == 12  # 3 in 4 bits is '0011', inverted is '1100' which is 12

# Test edge cases for _invert_integer_binary_representation
def test_edge_cases():
    assert _invert_integer_binary_representation(1) == 1  # 1 in binary is '1', inverted is '1' which is 1
    assert _invert_integer_binary_representation(0) == 0  # 0 in binary is '0', inverted is '0' which is 0
    assert _invert_integer_binary_representation(255, 8) == 255  # 255 in 8 bits is '11111111', inverted is '11111111' which is 255
    assert _invert_integer_binary_representation(256, 9) == 1  # 256 in 9 bits is '100000000', inverted is '000000001' which is 1


if __name__ == "__main__":
    pytest.main(['-V', __file__])
