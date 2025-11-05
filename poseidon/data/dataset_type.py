from enum import Enum


class DatasetType(Enum):
    NF_BOT_IOT_V3 = "NF-BoT-IoT-v3"
    NF_CICIDS2018_V3 = "NF-CICIDS2018-v3"
    NF_TON_IOT_V3 = "NF-ToN-IoT-v3"
    NF_UNSW_NB15_V3 = "NF-UNSW-NB15-v3"
    CUSTOM = "custom"


__all__ = ['DatasetType']