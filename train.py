from DatasetManager.Participant import Participant

from DatasetManager.Enums.Day import Day
from DatasetManager.Enums.MileStone import MileStone

from Dataset.PatientDataset import FatigueDMODataset


def main():
    participant = Participant(10376)

    csv_filter_func = participant.ml.filter_csv_data
    participant_metadata = csv_filter_func(
        participant.ml.metadata, "Local.Participant", "==", 10376
    )
    nan_removed = csv_filter_func(
        participant_metadata, "MFISTO1N", ">", 0
    )

    print(nan_removed)

    print(participant.get_csv_column(nan_removed, "MFISTO1N"))

    dl = FatigueDMODataset([], 10376)

if __name__ == "__main__":
    main()
