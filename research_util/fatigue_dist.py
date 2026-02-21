from src.patient_data_dispatcher import PatientDataDispatcher, PatientDataType
from src.core.enums import MileStone

import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import os

all_fatigue_values = []
for milestone in MileStone:
    if milestone == MileStone.DMO:
        continue

    pdd = PatientDataDispatcher("config/config.yaml", [], milestone)
    metadata = pdd.get_patient_data(PatientDataType.META)
    # print(metadata)
    fatigue_values = metadata["MFISTO1N"].tolist()
    fatigue_values = list(filter(lambda x: x > 0, fatigue_values))

    all_fatigue_values += [*fatigue_values]
    
print(f"number of values: {len(all_fatigue_values)}")
print(f"max: {np.max(all_fatigue_values)}")
print(f"min: {np.min(all_fatigue_values)}")

normal = all_fatigue_values
uniform = stats.rankdata(all_fatigue_values) / len(all_fatigue_values) * 84

plt.hist(uniform)
plt.ylabel("Frequency")
plt.xlabel("Fatigue values")
plt.savefig(os.path.join(os.getcwd(), "research_util", "fatigue_distribution_uniform"))
plt.show()