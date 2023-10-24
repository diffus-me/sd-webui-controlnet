from enum import Enum
from typing import Optional, Union, Tuple, Dict

import numpy as np

InputImage = Union[np.ndarray, str]
InputImage = Union[Dict[str, InputImage], Tuple[InputImage, InputImage], InputImage]


class ResizeMode(Enum):
    """
    Resize modes for ControlNet input images.
    """

    RESIZE = "Just Resize"
    INNER_FIT = "Crop and Resize"
    OUTER_FIT = "Resize and Fill"

    def int_value(self):
        if self == ResizeMode.RESIZE:
            return 0
        elif self == ResizeMode.INNER_FIT:
            return 1
        elif self == ResizeMode.OUTER_FIT:
            return 2
        assert False, "NOTREACHED"


class ControlMode(Enum):
    """
    The improved guess mode.
    """

    BALANCED = "Balanced"
    PROMPT = "My prompt is more important"
    CONTROL = "ControlNet is more important"


class ControlNetUnit:
    """
    Represents an entire ControlNet processing unit.
    """

    def __init__(
            self,
            enabled: bool = True,
            module: Optional[str] = None,
            model: Optional[str] = None,
            weight: float = 1.0,
            image: Optional[InputImage] = None,
            resize_mode: Union[ResizeMode, int, str] = ResizeMode.INNER_FIT,
            low_vram: bool = False,
            processor_res: int = -1,
            threshold_a: float = -1,
            threshold_b: float = -1,
            guidance_start: float = 0.0,
            guidance_end: float = 1.0,
            pixel_perfect: bool = False,
            control_mode: Union[ControlMode, int, str] = ControlMode.BALANCED,
            **_kwargs,
    ):
        self.enabled = enabled
        self.module = module
        self.model = model
        self.weight = weight
        self.image = image
        self.resize_mode = resize_mode
        self.low_vram = low_vram
        self.processor_res = processor_res
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.pixel_perfect = pixel_perfect
        self.control_mode = control_mode

    def __eq__(self, other):
        if not isinstance(other, ControlNetUnit):
            return False

        return vars(self) == vars(other)
