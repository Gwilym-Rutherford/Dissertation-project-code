from collections import namedtuple
from .helper.enum_def import Site
from .types import CSVData, Ids

import pandas as pd
import operator
import yaml
import torch

class BaseLoader:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)

    def load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)
    
    def filter_by_site(
        self, site: Site, ids: Ids | None = None
    ) -> torch.Tensor:
        if ids is None:
            ids = self.get_all_ids()
        prefixes = ids.div(1000).floor()
        mask = prefixes == int(site.value)
        return ids[mask]


    def filter_csv_data(
        self, data: CSVData, *conditions: namedtuple
    ) -> pd.DataFrame | None:
        ops = {
            ">": operator.gt,
            "<": operator.lt,
            "==": operator.eq,
            "!=": operator.ne,
        }

        for condition in conditions:
            mask = ops[condition.operation](data[condition.column], condition.value)
            processed_data = data[mask]
            if processed_data.empty:
                return None
            data = processed_data

        return data

    def filter_csv_column(self, data: CSVData, columns: list[str], keep_id = False) -> CSVData | None:
        
        if keep_id:
            columns.append("participant_id")
            columns.append("Local.Participant")
        
        columns = set(columns)
        
        mask = [col for col in columns if col in data.columns]
        return data[mask]
