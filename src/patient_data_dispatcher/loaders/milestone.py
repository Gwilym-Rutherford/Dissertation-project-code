# gets all patient data grouped by milestone instead of just aggregated like in dmo.py
# default behaviour is to just use all dmo features
from .base import BaseLoader
from src.core.types import DMOFeatures, ListIds, CSVData
from src.core.enums import MileStone
from src.core.data_transforms import Transform

import pandas as pd
import numpy as np
import torch
import os
import re


class MileStoneLoader(BaseLoader):
    def __init__(
        self,
        config_path: str,
        milestone: MileStone,
        metadata: CSVData,
        dmo_features: DMOFeatures = None,
    ) -> None:
        super().__init__(config_path)
        self.path = self.config["paths"]["dmo_data_path"]
        self.metadata = metadata

        if milestone == MileStone.ALL:
            self.milestone = [
                MileStone.T1,
                MileStone.T2,
                MileStone.T3,
                MileStone.T4,
                MileStone.T5,
            ]
        else:
            self.milestone = [milestone]

        if dmo_features is not None:
            self.dmo_features = dmo_features.copy()
        else:
            self.dmo_features = None

    def __call__(self, ids: ListIds) -> CSVData | tuple[torch.Tensor, torch.Tensor]:
        return self.get_dmo_data(ids)

    def get_dmo_data(self, ids: ListIds) -> CSVData | tuple[torch.Tensor, torch.Tensor]:

        dir_list = os.listdir(self.path)

        dmo_data_by_milestone = []

        # get respected milstone data
        for milestone in self.milestone:
            milestone_dir = [x for x in dir_list if x[:2] == milestone.value][0]
            milestone_dir_list = os.listdir(os.path.join(self.path, milestone_dir))
            csv_file = [
                x for x in milestone_dir_list if re.match(r".*daily_agg_all.*.csv$", x)
            ][0]
            csv_path = os.path.join(self.path, milestone_dir, csv_file)

            reader = pd.read_csv(csv_path, chunksize=100000, low_memory=False)
            dmo_reader = pd.concat(reader, ignore_index=True)

            dmo_data_by_milestone.append(dmo_reader)

        if ids is None:
            print("get_dmo_data() needs a list or set of ids")
            quit(-1)

        # filter data by ids
        dmo_data_by_milestone = [
            df[df["participant_id"].isin(ids)] for df in dmo_data_by_milestone
        ]

        dmo_data_all_milestones = pd.concat(dmo_data_by_milestone)

        if not self.dmo_features:
            self.dmo_features = dmo_data_all_milestones.columns.remove(
                "measurement_date"
            )
        else:
            self.dmo_features = ["participant_id", "visit_type"] + self.dmo_features

        dmo_data_all_milestones = dmo_data_all_milestones[self.dmo_features]

        # patient input shape [patients, milestones, days, features]
        patient_input = []
        for patient in dmo_data_all_milestones.groupby(["participant_id"]):
            milestones = patient[1].groupby(["visit_type"])

            patient_milestones = []
            for visit in milestones:
                visit = visit[1].head(7).drop(["participant_id", "visit_type"], axis=1)
                visit = visit.fillna(-1)
                visit = visit.replace("nan", -1)
                visit_mat = visit.to_numpy()

                rows, cols = visit_mat.shape
                if rows < 7:
                    missing_milestones = 7 - rows
                    padding = np.full((missing_milestones, cols), fill_value=-1)
                    visit_mat = np.concatenate((visit_mat, padding), axis=0)

                patient_milestones.append(visit_mat)

            patient_milestones = np.array(patient_milestones)
            milestones, rows, cols = patient_milestones.shape

            if milestones < 5:
                missing_milestones = 5 - milestones
                padding = np.full((missing_milestones, rows, cols), fill_value=-1)
                patient_milestones = np.concatenate(
                    (patient_milestones, padding), axis=0
                )

            patient_input.append(patient_milestones)
        patient_input = torch.tensor(np.array(patient_input))

        patient_labels = []

        milestones = [
            m.value.lower() for m in MileStone if m.value.lower() not in ["dmos", "all"]
        ]

        for patient_id, patient_df in dmo_data_all_milestones.groupby("participant_id"):
            milestone_values = []
            for milestone in milestones:
                subset = self.metadata[
                    (self.metadata["Local.Participant"] == patient_id)
                    & (self.metadata["visit.number"] == milestone)
                ]

                if len(subset) == 1:
                    value = subset["MFISTO1N"].iloc[0]
                    if pd.isna(value):
                        value = -1
                else:
                    value = -1

                milestone_values.append([value])
            patient_labels.append(milestone_values)
        patient_labels = torch.tensor(np.array(patient_labels))

        return patient_input, patient_labels
