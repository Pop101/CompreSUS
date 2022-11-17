import numpy as np
from numba import njit, jit

# returns the average color of each channel of the image
# a numpy array of shape img.shape[:-1]
@njit
def avg_color(image: np.array):
    image = image.copy()
    color_sum = image.reshape(-1, image.shape[-1]).sum(axis=0)
    if color_sum[-1] != 0:
        return color_sum / color_sum[-1]
    return np.zeros(image.shape[-1])


# method to get the best color for an amogus
# returns the average color of each channel of the image
@njit
def match_block(block, img):
    filtered = img * block
    return avg_color(filtered)


# method to rate a block
@njit
def rate_match(color, image):
    return np.sum(np.abs(color - avg_color(image)))


# Multiplies each channel in the image by the color
@njit
def colorize(image, color: list):
    image = image.copy()
    image[:, :, 0] *= color[0]
    image[:, :, 1] *= color[1]
    image[:, :, 2] *= color[2]
    return image

@njit
def match_pattern(pattern, img):
    inverse_pattern = np.ones(pattern.shape) - pattern

    pattern_color = match_block(pattern, img)
    pattern_flip_color = match_block(np.fliplr(pattern), img)

    # Rate both the pattern and the flipped pattern
    should_flip = rate_match(pattern_flip_color, img) < rate_match(pattern_color, img)
    pattern_color = pattern_flip_color if should_flip else pattern_color

    # Perform the same check on the inverse pattern
    inverse = inverse_pattern if not should_flip else np.fliplr(inverse_pattern)
    inverse_color = match_block(inverse, img)

    return (should_flip, pattern_color, inverse_color)



def pad(img, before: tuple = None, after: tuple = None):
    if before == None:
        before = np.zeros((len(img.shape) - 1,), dtype=np.int)
    if after == None:
        after = np.zeros((len(img.shape) - 1,), dtype=np.int)

    img = np.pad(
        img,
        pad_width=np.array([(b, a) for b, a in zip(before, after)] + [(0, 0)]),
        mode="constant",
        constant_values=0,
    )
    return img


def chop(img, before: tuple = None, after: tuple = None):
    if before == None:
        before = (0, 0)
    if after == None:
        after = (0, 0)

    new_end = (img.shape[0] - after[0], img.shape[1] - after[1])

    return img[before[0] : new_end[0], before[1] : new_end[1]]


def calc_remainders(shape, offset):
    return [abs(shape[x] - offset[x]) % shape[x] for x, _ in enumerate(offset)]


def calc_steps(shape, offset):
    return [(shape[x] + offset[x] - 1) // shape[x] for x, _ in enumerate(offset)]
