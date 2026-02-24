from src.patient_data_dispatcher import PatientDataDispatcher, PatientDataType
from src.core.enums import MileStone

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_distribution(fatigue_values: list[int], filename: str):
    print(f"number of values: {len(fatigue_values)}")
    print(f"max: {np.max(fatigue_values)}")
    print(f"min: {np.min(fatigue_values)}")

    plt.hist(fatigue_values)
    plt.ylabel("Frequency")
    plt.xlabel("Fatigue values")
    plt.savefig(os.path.join(os.getcwd(), "src", "research_util", f"{filename}"))
    plt.show()

# # All fatigue values
# all_fatigue_values = []
# for milestone in MileStone:
#     if milestone == MileStone.DMO:
#         continue

#     pdd = PatientDataDispatcher("config/config.yaml", [], milestone)
#     metadata = pdd.get_patient_data(PatientDataType.META)
#     # print(metadata)
#     fatigue_values = metadata["MFISTO1N"].tolist()
#     fatigue_values = list(filter(lambda x: x > 0, fatigue_values))

#     all_fatigue_values += [*fatigue_values]

# plot_distribution(all_fatigue_values, "fatigue_distribution")