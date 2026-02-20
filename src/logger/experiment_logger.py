from math import ceil
from datetime import datetime
from .model_config import ModelConfig
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import os
import json


class ExperimentLogger:
    def __init__(self, config: ModelConfig, output_dir: str = "log output"):
        self.config = config

        self.output_dir = os.path.join(os.getcwd(), output_dir)
        self.graphs = {}

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def log_values(self, values: list(tuple[str, float])) -> None:
        for key, value in values:
            if key not in self.graphs.keys():
                self.graphs[key] = []
            self.graphs[key].append(value)

    def save(self, accuracy: float, show_fig: bool = True):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        dir_path = os.path.join(self.output_dir, self.config.name, timestamp)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, )

        plt = self._make_graph()
        plt.savefig(os.path.join(dir_path, "plot"))

        config_dict = self.config.__dict__
        config_dict["accuracy"] = f"{accuracy:.2f}%"

        with open(os.path.join(dir_path, "config.json"), "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

        if show_fig:
            plt.show()

    def _make_graph(self) -> Figure:
        n_cols = 2
        n_rows = ceil(len(self.graphs.keys()) / n_cols)

        plt.figure(figsize=(15, n_rows * 4))
        for index, key in enumerate(self.graphs):
            plt.subplot(n_rows, n_cols, index + 1)

            y_values = self.graphs[key]
            x_values = list(range(0, len(y_values)))

            plt.plot(
                x_values,
                y_values,
            )

            plt.title(f"{self.config.name}-{key}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")

            plt.ylim(bottom=0)
            plt.grid()

        plt.figtext(
            0.5, 0.01, self.config.notes, wrap=True, horizontalalignment="center", fontsize=10
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        return plt
