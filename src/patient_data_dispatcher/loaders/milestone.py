# gets all patient data grouped by milestone instead of just aggregated like in dmo.py
# default behaviour is to just use all dmo features
from .base import BaseLoader
from src.core.types import DMOFeatures, ListIds, CSVData
from src.core.enums import MileStone, DataFrequency
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
        data_frequency: DataFrequency,
        filtered: bool,
        dmo_features: DMOFeatures = None,
        static_features: list[str] = None,
    ) -> None:
        super().__init__(config_path)
        self.path = self.config["paths"]["dmo_data_path"]
        self.metadata = metadata
        self.data_frequency_string = data_frequency.value
        self.data_frequency = data_frequency
        self.filtered = "filtered" if filtered else "all"
        self.static_features = static_features

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

    def _get_dmo_data_by_day(self, dmo_data_all_milestones):

        # gets updated later
        n_columns = 0

        patient_input = []
        for patient in dmo_data_all_milestones.groupby(["participant_id"]):
            milestones = patient[1].groupby(["visit_type"])

            patient_milestones = []
            milestone_dates = []
            for visit in milestones:
                visit = visit[1].head(7).drop(["participant_id", "visit_type"], axis=1)
                visit = visit.fillna(-1)
                visit = visit.replace("nan", -1)

                n_columns = len(visit.columns)

                visit["measurement_date"] = pd.to_datetime(visit["measurement_date"])

                group = visit.set_index("measurement_date")
                start_date = group.index.min()
                milestone_dates.append(start_date)
                date_range = pd.date_range(start_date, periods=7, freq="D")

                visit = group.reindex(date_range, fill_value=-1).reset_index(drop=True)

                patient_milestones.append(visit.to_numpy())

            # if there are less than 5 milestones, find out which ones are missing
            if len(milestone_dates) < 5:
                updated_dates = [milestone_dates[0]]
                prev_date = milestone_dates[0]
                for date in milestone_dates[1:]:
                    curr_date = date

                    num_months = (curr_date.year - prev_date.year) * 12 + (
                        curr_date.month - prev_date.month
                    )

                    if num_months > 7:
                        n_missed_visits = (
                            int(np.floor(round((num_months / 6) * 6) / 6)) - 1
                        )

                        for missed in range(n_missed_visits):
                            updated_dates.append(-1)

                    updated_dates.append(curr_date)
                    prev_date = curr_date

                if len(updated_dates) < 5:
                    missing_end_visits = 5 - len(updated_dates)
                    for missed in range(missing_end_visits):
                        updated_dates.append(-1)

                milestone_dates = updated_dates

            # fill in missing milestones with -1
            updated_patient_milestones = []
            milestone_index = 0
            for date in milestone_dates:
                if date != -1:
                    updated_patient_milestones.append(
                        patient_milestones[milestone_index]
                    )
                    milestone_index += 1
                else:
                    updated_patient_milestones.append(
                        np.full((7, n_columns - 1), fill_value=-1)
                    )

            patient_input.append(np.array(updated_patient_milestones))

        patient_input = torch.tensor(np.array(patient_input))

        return patient_input

    def _get_dmo_data_by_week(self, dmo_data_all_milestones):
        # patient input shape [patients, milestones, features]
        patient_input = []
        for patient in dmo_data_all_milestones.groupby(["participant_id"]):
            milestones = patient[1].groupby(["visit_type"])

            patient_features = []
            for visit in milestones:
                visit = visit[1].head(5).drop(["participant_id", "visit_type"], axis=1)
                visit = visit.fillna(-1)
                visit = visit.replace("nan", -1)
                visit_mat = visit.to_numpy()
                patient_features.append(np.squeeze(visit_mat))

            patient_features = np.array(patient_features)

            milestones, rows = patient_features.shape

            if milestones < 5:
                missing_milestones = 5 - milestones
                padding = np.full((missing_milestones, rows), fill_value=-1)
                patient_features = np.concatenate((patient_features, padding), axis=0)

            patient_input.append(patient_features)

        patient_input = np.array(patient_input)
        return torch.tensor(patient_input)

    def get_dmo_data(self, ids: ListIds) -> CSVData | tuple[torch.Tensor, torch.Tensor]:

        dmo_data_all_milestones = self.get_plain_milestone_data(ids)
        if self.static_features:
            static_features = self.get_static_features()

            dmo_data_all_milestones = dmo_data_all_milestones.merge(
                static_features,
                left_on="participant_id",
                right_on="Local.Participant",
                how="left",
            )
            dmo_data_all_milestones = dmo_data_all_milestones.drop(
                columns=["Local.Participant"]
            )

        if self.dmo_features:
            self.dmo_features = [
                "participant_id",
                "visit_type",
                "measurement_date",
            ] + self.dmo_features
            if self.static_features:
                self.dmo_features = self.dmo_features + self.static_features

        # select columns matching dmo features for either week or daily
        selected_cols = [
            col
            for col in dmo_data_all_milestones.columns
            if any(col.startswith(p) for p in self.dmo_features)
        ]

        dmo_data_all_milestones = dmo_data_all_milestones[selected_cols]

        # get appropriate dmo data and format for either week or daily
        if self.data_frequency == DataFrequency.DAILY:
            patient_input = self._get_dmo_data_by_day(dmo_data_all_milestones)
        else:
            patient_input = self._get_dmo_data_by_week(dmo_data_all_milestones)

        # get labels for all patients
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

    def get_plain_milestone_data(self, ids: ListIds) -> CSVData:
        dir_list = os.listdir(self.path)
        dmo_data_by_milestone = []

        # get respected milestone data
        for milestone in self.milestone:
            milestone_dir = [x for x in dir_list if x[:2] == milestone.value][0]
            milestone_dir_list = os.listdir(os.path.join(self.path, milestone_dir))

            regex_ex = rf".*{self.data_frequency_string}_agg_{self.filtered}.*.csv$"

            csv_file = [x for x in milestone_dir_list if re.match(regex_ex, x)][0]
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
        return dmo_data_all_milestones

    def get_static_features(self) -> pd.DataFrame:
        features = self.metadata[["Local.Participant", *self.static_features]]
        patient_averages = features.groupby("Local.Participant").mean(numeric_only=True)
        patient_averages = patient_averages.reset_index()

        return patient_averages
