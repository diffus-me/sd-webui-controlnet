from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Tuple, Dict, List

import numpy as np

from scripts.enums import HiResFixOption


class ControlMode(Enum):
    """
    The improved guess mode.
    """

    BALANCED = "Balanced"
    PROMPT = "My prompt is more important"
    CONTROL = "ControlNet is more important"


class BatchOption(Enum):
    DEFAULT = "All ControlNet units for all images in a batch"
    SEPARATE = "Each ControlNet unit for each image in a batch"


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


InputImage = Union[np.ndarray, str]
InputImage = Union[Dict[str, InputImage], Tuple[InputImage, InputImage], InputImage]


@dataclass
class ControlNetUnit:
    """
    Represents an entire ControlNet processing unit.
    """
    enabled: bool = True
    module: str = "none"
    model: str = "None"
    weight: float = 1.0
    image: Optional[Union[InputImage, List[InputImage]]] = None
    resize_mode: Union[ResizeMode, int, str] = ResizeMode.INNER_FIT
    low_vram: bool = False
    processor_res: int = -1
    threshold_a: float = -1
    threshold_b: float = -1
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    pixel_perfect: bool = False
    control_mode: Union[ControlMode, int, str] = ControlMode.BALANCED
    # Whether to crop input image based on A1111 img2img mask. This flag is only used when `inpaint area`
    # in A1111 is set to `Only masked`. In API, this correspond to `inpaint_full_res = True`.
    inpaint_crop_input_image: bool = True
    # If hires fix is enabled in A1111, how should this ControlNet unit be applied.
    # The value is ignored if the generation is not using hires fix.
    hr_option: Union[HiResFixOption, int, str] = HiResFixOption.BOTH

    # Whether save the detected map of this unit. Setting this option to False prevents saving the
    # detected map or sending detected map along with generated images via API.
    # Currently the option is only accessible in API calls.
    save_detected_map: bool = True

    # Weight for each layer of ControlNet params.
    # For ControlNet:
    # - SD1.5: 13 weights (4 encoder block * 3 + 1 middle block)
    # - SDXL: 10 weights (3 encoder block * 3 + 1 middle block)
    # For T2IAdapter
    # - SD1.5: 5 weights (4 encoder block + 1 middle block)
    # - SDXL: 4 weights (3 encoder block + 1 middle block)
    # Note1: Setting advanced weighting will disable `soft_injection`, i.e.
    # It is recommended to set ControlMode = BALANCED when using `advanced_weighting`.
    # Note2: The field `weight` is still used in some places, e.g. reference_only,
    # even advanced_weighting is set.
    advanced_weighting: Optional[List[float]] = None

    def __eq__(self, other):
        if not isinstance(other, ControlNetUnit):
            return False

        return vars(self) == vars(other)

    def accepts_multiple_inputs(self) -> bool:
        """This unit can accept multiple input images."""
        return self.module in (
            "ip-adapter_clip_sdxl",
            "ip-adapter_clip_sdxl_plus_vith",
            "ip-adapter_clip_sd15",
            "ip-adapter_face_id",
            "ip-adapter_face_id_plus",
            "instant_id_face_embedding",
        )

    def to_args(self) -> list:
        return [
            self.enabled,
            self.module,
            self.model,
            self.weight,
            self.image,
            self.resize_mode,
            self.low_vram,
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.guidance_start,
            self.guidance_end,
            self.pixel_perfect,
            self.control_mode,
            self.inpaint_crop_input_image,
            self.hr_option,
        ]

