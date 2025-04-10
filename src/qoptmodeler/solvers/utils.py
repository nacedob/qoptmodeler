import numpy as np


def bits_to_int(bits: [list, np.ndarray], reverse=False) -> int:
    """
    Convert a list or numpy array of bits to an integer.

    Parameters:
    bits (list or np.ndarray): Sequence of bits (0s and 1s) to convert.
    reverse (bool, optional): If True, reverses the order of bits before conversion. Default is False.

    Returns:
    int: The integer representation of the binary sequence.
    """
    bits = list(bits)  # Ensure it's a list
    if reverse:
        bits = bits[::-1]
    return int("".join(map(str, bits)), 2)


def int_to_bits(n: int, length: int = None, reverse: bool = False) -> np.ndarray:
    """
    Convert an integer to a list of bits.

    Parameters:
    n (int): The integer to convert.
    length (int, optional): If provided, pads the bit representation to this length with leading zeros.
    reverse (bool, optional): If True, reverses the order of bits after conversion. Default is False.

    Returns:
    np.ndarray: The binary representation of the integer as an array of bits.
    """
    bits = list(map(int, bin(n)[2:]))  # Convert to binary and extract digits
    if length:
        bits = [0] * (length - len(bits)) + bits  # Pad with leading zeros if needed
    if reverse:
        bits = bits[::-1]
    return np.array(bits)


def invert_integer_binary_representation(n: int, num_bits: int = None) -> int:
    """
    Inverts the bits of an integer n, considering a fixed number of bits.

    Args:
        n (int): The integer whose bits are to be inverted.
        num_bits (int, optional): The number of bits to consider in the inversion. Default to n.bit_length()

    Returns:
        int: The integer obtained after inverting the bits of n.

    Example:
        invert_bits(6, 4) returns 3 because 6 in 4-bit binary is '0110', and its inversion is '0110' = 3.
    """
    if not num_bits:
        num_bits = n.bit_length()

    # Convert n to its binary representation with the specified number of bits
    binary = bin(n)[2:].zfill(num_bits)  # Pad with leading zeros if needed

    # If the binary number has more bits than specified, truncate it
    binary = binary[-num_bits:]

    # Invert the binary string
    inverted_binary = binary[::-1]

    # Convert the inverted binary string back to an integer
    N_inverted = int(inverted_binary, 2)

    return N_inverted


def to_bitstring(integer: int, num_bits: int) -> list[int]:
    return [int(digit) for digit in f"{integer:0{num_bits}b}"]


def from_optimization_to_qubo(qubo_quad: np.ndarray,
                              qubo_linear: np.ndarray,
                              eq_constraints_lhs: list[np.ndarray],
                              eq_constraints_rhs: list[np.ndarray]) -> np.ndarray:
    """
    Tiene que devolver la matriz que define el qubo
    :param qubo_quad:
    :param qubo_linear:
    :param eq_constraints_lhs:
    :param eq_constraints_rhs:
    :return:
    """
    raise NotImplemented()



class MaxTimeWarning(Warning):
    pass


