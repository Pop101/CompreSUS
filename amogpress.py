from io import BytesIO
from skimage import io
from skimage.util.shape import view_as_blocks
import numpy as np
from PIL import Image

from deflate import gzip_compress as compress
from deflate import gzip_decompress as decompress

from utils.encoding import *
from utils.imgutils import *
from utils.patterns import *
from utils.fileutils import *

# Read and normalize image to 0.0-1.0 range
compression = bytearray()


def encode_image(img_path, out_path):
    global compression
    img = io.imread(img_path) / 255

    # If only 3 channels, add an alpha channel
    if img.shape[-1] == 3:
        img = np.concatenate([img, np.ones((img.shape[0], img.shape[1], 1))], axis=2)

    # Pad output image to match amogus shape
    # Calculate size so that the image can be divided into amogus size blocks
    width_remainder = AMOGI.shape[0] - img.shape[0] % AMOGI.shape[0]
    height_remainder = AMOGI.shape[1] - img.shape[1] % AMOGI.shape[1]
    img = pad(img, (width_remainder, height_remainder))

    # Copy original image as a fallback
    org_img = img.copy()

    # Encode metadata
    compression = bytearray()
    compression.extend(np.uint32(img.shape[0] // AMOGI.shape[0]).tobytes())
    compression.extend(np.uint32(img.shape[1] // AMOGI.shape[1]).tobytes())

    offsets = [(0, 0), (2, 2)]
    for offset in offsets:
        offset_remainder = calc_remainders(AMOGI.shape, offset)

        # Crop image if offset is set
        img = chop(org_img, offset_remainder, offset)

        # Split image into blocks
        amog_blocks = view_as_blocks(img, block_shape=AMOGI.shape)

        for a_x, a_y in np.ndindex(amog_blocks.shape[:2]):
            block = amog_blocks[a_x, a_y, 0]

            # Note: encoding the flip bit twice is not necessary
            # However, it keeps the code cleaner
            flip, color, inverse_color = match_pattern(AMOGI, block)
            pattern = encode_pattern(flip, color)
            compression.extend(pattern)

            # Only encode the background color for the first iteration
            if offset == (0, 0):
                inverse = encode_pattern(flip, inverse_color)
                compression.extend(inverse)

    with open(out_path, "wb") as f:
        f.write(compress(compression, 10))


# Convert compressed data back to image
def decode_image(compressed_path, out_path, sussyness: float = 0.0):
    with open(compressed_path, "rb") as f:
        compressed = decompress(f.read())

    # Generate patterns used
    PATTERN = AMOGI
    PATTERN_FLIP = np.flip(PATTERN, axis=1)
    PATTERN_INVERSE = np.ones(PATTERN.shape) - PATTERN
    PATTERN_INVERSE_FLIP = np.ones(PATTERN.shape) - PATTERN_FLIP

    # Decode metadata
    stream = BytesIO(compressed)
    width = np.frombuffer(stream.read(4), dtype=np.uint32)[0]
    height = np.frombuffer(stream.read(4), dtype=np.uint32)[0]

    reconstructed_image = Image.new(
        "RGBA", (height * PATTERN.shape[1], width * AMOGI.shape[0])
    )

    offsets = [(0, 0), (2, 2)]
    for offset in offsets:
        x_diff, y_diff = calc_steps(PATTERN.shape, offset)
        img_part = np.zeros(
            (
                (width - x_diff) * PATTERN.shape[0],
                (height - y_diff) * PATTERN.shape[1],
                4,
            )
        )
        recon_blocks = view_as_blocks(img_part, block_shape=PATTERN.shape)

        for r_x, r_y in np.ndindex(recon_blocks.shape[:2]):
            flip, color = decode_pattern(stream)

            _pattern = PATTERN if not flip else PATTERN_FLIP
            _pattern = colorize(_pattern, color)

            if not x_diff and not y_diff:
                _, inverse_color = decode_pattern(stream)
                inverse = PATTERN_INVERSE if not flip else PATTERN_INVERSE_FLIP
                inverse = colorize(inverse, (1.0 - sussyness) * inverse_color)
                _pattern = np.clip(_pattern + inverse, 0, 1)

            recon_blocks[r_x, r_y, 0] = _pattern

        # Pad the image part by the offset
        if x_diff or y_diff:
            img_part = pad(
                img_part,
                offset,
                (PATTERN.shape[0] - offset[0], PATTERN.shape[1] - offset[1]),
            )

        # Compose the image
        img_part = Image.fromarray(img_part.__mul__(255).astype(np.uint8))
        reconstructed_image = Image.alpha_composite(reconstructed_image, img_part)

    # Crop out the border row
    reconstructed_image = reconstructed_image.crop(
        np.array(reconstructed_image.getbbox())
        + 1 * np.array([AMOGI.shape[1], AMOGI.shape[0], 0, 0])
    )

    # Save the result
    reconstructed_image.save(out_path)


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--encode", help="Path to image to encode")
    parser.add_argument("-d", "--decode", help="Path to compressed data to decode")
    parser.add_argument("-o", "--output", help="Path to output file")
    parser.add_argument(
        "-s", "--sus", help="Sussiness of the image", type=float, default=0.1
    )

    args = parser.parse_args()

    if not args.output or "." not in str(args.output):
        args.output = set_extension(args.output or args.encode or args.decode, "amog")

    if args.encode:
        print("Encoding image...")
        encode_image(args.encode, args.output)
        decode_image(args.output, args.output + ".png", sussyness=args.sus)
    elif args.decode:
        print("Decoding amogimage...")
        decode_image(args.decode, args.output, sussyness=args.sus)
    else:
        print("Please specify a file to either --encode or --decode")
        sys.exit(1)
