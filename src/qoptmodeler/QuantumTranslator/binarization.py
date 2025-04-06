import numpy as np


def continuous_to_binary(value, num_bits=8, min_val=0, max_val=1):
    """
    Encodes a continuous variable into binary variables (bitwise encoding).

    Parameters:
    - value (float): The continuous variable to encode.
    - num_bits (int): The number of binary variables to use for encoding.
    - min_val (float): The minimum value of the continuous variable range.
    - max_val (float): The maximum value of the continuous variable range.

    Returns:
    - list: A list of binary values representing the continuous variable.
    """
    # Normalize the value to the range [0, 1]
    normalized_value = (value - min_val) / (max_val - min_val)

    # Convert to an integer in the range [0, 2^num_bits - 1]
    encoded_value = int(normalized_value * (2 ** num_bits - 1))

    # Convert the integer to a binary representation
    binary_encoding = np.array([int(x) for x in np.binary_repr(encoded_value, width=num_bits)])

    return binary_encoding


# Example usage
value = 0.5
binary_value = continuous_to_binary(value)
print(binary_value)
