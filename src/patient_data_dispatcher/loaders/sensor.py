from .base import BaseLoader

class SensorLoader(BaseLoader):
    def __init__(self, config_path):
        super().__init__(config_path)

    def __call__(self, ids):
        pass

    