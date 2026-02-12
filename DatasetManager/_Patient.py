from dataclasses import dataclass
from DatasetManager.helper.named_tuple_def import RawData
from DatasetManager.helper.enum_def import MileStone

import torch
import pandas


@dataclass
class _Patient:
    id: int
    meta_data: pandas.DataFrame | None
    sensor_dmo_data: torch.Tensor | None
    sensor_raw_data: dict[str, dict[str, RawData]] | None

    def get_fatigue_at_milestone(self, milestone: MileStone) -> float:
        mask = self.meta_data["visit.number"] == milestone.value.lower()
        return self.meta_data[mask]['MFISTO1N'].item()
        
        
