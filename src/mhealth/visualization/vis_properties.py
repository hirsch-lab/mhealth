import string
from dataclasses import dataclass

from ..utils.data_aggregator import Normalization


@dataclass
class VisProperties:
    in_dir: str
    out_dir: str
    keys: dict
    min_scale: float
    max_scale: float
    start_idx: int
    end_idx: int
    short_keys: dict
    colormap: str = "coolwarm"
    fill_hours: bool = True
    normalization: Normalization=Normalization.NONE
