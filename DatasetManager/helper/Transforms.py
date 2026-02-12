from DatasetManager.helper.enum_def import MileStone, Day

import pandas as pd
import torch


class Transforms():
    @staticmethod
    def none_type(data) -> torch.Tensor:
        return torch.zeros((len(MileStone) - 1) * len(Day), 5)