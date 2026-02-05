from DatasetManager.helper.Helper import Helper

import pandas as pd
import os
import operator


class DataLoader(object):
    def __init__(self):
        self.ops = {
            ">": operator.gt,
            "<": operator.lt,
            "==": operator.eq,
            "!=": operator.ne,
        }
        self.curr_file_path = os.path.dirname(os.path.abspath(__file__))
        
        self.config = Helper.load_config_yaml(
            os.path.join(os.getcwd(), "config/config.yaml")
        )

    def create_csv_data(self, csv_path):
        reader = pd.read_csv(csv_path, chunksize=100000, low_memory=False)
        data = pd.concat(reader, ignore_index=True)
        return data

    def filter_csv_data(self, data, column, condition, value):
        mask = self.ops[condition](data[column], value)
        return data[mask]

    def get_csv_column(self, csv_data, column_name):
        return csv_data[column_name].to_list()

    def filter_json_arr(self, json_arr, attr, condition, value):
        df = pd.DataFrame(json_arr)
        mask = self.ops[condition](df[attr], value)
        return df[mask].to_dict(orient="records")

    def path_not_exist(self, path):
        print(f"no path found, make sure {path} exists!")
        quit(1)

    def check_empty(self, data, data_type="data"):
        if data.empty:
            print(f"{data_type} does not exist!")
            return None
        return data

    def show_dir_contents(self, path):
        print(os.listdir(path))
