import numpy as np

AMOGI = np.array(
    [
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ]
)
AMOGI = np.stack([AMOGI] * 4, axis=-1)
INVERSE_AMOGI = np.ones(AMOGI.shape) - AMOGI
