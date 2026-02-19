from dataclasses import dataclass
from .types import CSVData


@dataclass
class Patient:
    id: int
    meta_data: CSVData | None
    sensor_dmo_data: CSVData | None
    #sensor_dmo_data_tensor: torch.Tensor | None
    #sensor_raw_data: RawSensorData | None

    # def get_fatigue_at_milestone(self, milestone: MileStone) -> float:
    #     mask = self.meta_data["visit.number"] == milestone.value.lower()
    #     return self.meta_data[mask]['MFISTO1N'].item()