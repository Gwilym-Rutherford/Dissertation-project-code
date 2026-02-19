from abc import ABC, abstractmethod
import yaml

class BaseLoader(ABC):
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)

    def load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)

    @abstractmethod
    def __call__(self):
        pass
