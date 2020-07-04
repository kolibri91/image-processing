import cv2
import numpy as np
from typing import Iterable

from .block import Block


class Resize(Block):
    def __init__(self, factors: Iterable):
        self.factors = factors

    def __call__(self, input_image: np.ndarray) -> np.ndarray:
        return cv2.resize(input_image, (0, 0), None, self.factors[0], self.factors[1])
