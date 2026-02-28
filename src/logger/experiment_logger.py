from math import ceil
from datetime import datetime
from .model_config import ModelConfig
from matplotlib.figure import Figure
from src.research_util import compute_all_metrics, confusion_matrix

import matplotlib.pyplot as plt
import os
import torch
import json


class ExperimentLogger:
    def __init__(self, config: ModelConfig, output_dir: str = "log output") -> None:
        self.config = config

        self.output_dir = os.path.join(os.getcwd(), output_dir)
        self.graphs = {}

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def log_values(self, values: list[tuple[str, float]]) -> None:
        for key, value in values:
            if key not in self.graphs.keys():
                self.graphs[key] = []
            self.graphs[key].append(value)

    def save(self, pred: torch.tensor, labels: torch.tensor, show_fig: bool = True):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        dir_path = os.path.join(self.output_dir, self.config.name, timestamp)
        if not os.path.isdir(dir_path):
            os.makedirs(
                dir_path,
            )

        plt = self._make_graph()
        plt.savefig(os.path.join(dir_path, "plot"))

        confusion_plt = confusion_matrix(pred, labels)
        confusion_plt.savefig(os.path.join(dir_path, "confusion"))

        metrics = compute_all_metrics(pred, labels)

        config_dict = self.config.__dict__
        config_dict.update(metrics)
        config_dict["optimiser"] = f"{str(config_dict['optimiser'])}"
        config_dict["loss_fn"] = f"{str(config_dict['loss_fn'])}"

        with open(os.path.join(dir_path, "config.json"), "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

        if show_fig:
            plt.show()

    def _make_graph(self) -> Figure:
        n_cols = 2
        n_rows = ceil(len(self.graphs.keys()) / n_cols)

        all_values = [val for sublist in self.graphs.values() for val in sublist]
        global_max = max(all_values) if all_values else 1.0

        axes = None
        plt.figure(1, figsize=(15, n_rows * 8))
        for index, key in enumerate(self.graphs):
            ax = plt.subplot(n_rows, n_cols, index + 1, sharey=axes)

            if axes is None:
                axes = ax

            y_values = self.graphs[key]
            x_values = list(range(0, len(y_values)))

            plt.plot(
                x_values,
                y_values,
            )

            plt.title(f"{self.config.name}-{key}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")

            plt.ylim(0, global_max * 1.05)
            plt.grid()

        plt.figtext(
            0.5,
            0.01,
            self.config.notes,
            wrap=True,
            horizontalalignment="center",
            fontsize=10,
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        return plt
