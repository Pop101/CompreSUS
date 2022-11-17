from io import BytesIO
import numpy as np
from numba import njit

@njit(fastmath=True)
def encode_pattern(flip_bit: bool, color: list) -> np.uint32:
    # Convert each color to uint8
    color = [np.uint8(c * 255) for c in color]

    # Replace the LSB of blue with the flip bit
    color[2] &= 0b11111110
    color[2] |= 0b00000001 if flip_bit else 0b00000000

    # Merge all the bytes into a single uint32
    return np.uint32(color[0] << 16 | color[1] << 8 | color[2])
    
    
def decode_pattern(bytestream: BytesIO):
    # Read 3 numbers
    color = np.frombuffer(bytestream.read(3), dtype=np.uint8).copy()
    color = np.flip(color, axis=0)

    # Extract the flip bit from the blue channel
    flip_bit = (color[2] & 0b00000001) == 0b00000001

    # Remove the flip bit from the blue channel
    color[2] &= 0b11111110

    # Add alpha to color
    color = np.append(color, np.uint8(255))

    # Convert to floats
    color = color / 255
    return (flip_bit, color)
