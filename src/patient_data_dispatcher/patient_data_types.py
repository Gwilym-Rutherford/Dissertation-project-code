from enum import Enum

class PatientDataType(Enum):
    META = "meta"
    DMO = "dmo"
    SENSOR = "sensor"
    MILESTONE = "milestone"