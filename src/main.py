import pandas as pd
from pathlib import Path
from src.DataManager import DataManager
from src.FeatureManager import FeatureManager
from src.configuration import config
from src.utils import DfCustomPrintFormat

# Load data
data = DataManager()
data.LoadData()

# Feature Engineering
features = FeatureManager()
features.EngineerFeatures(data)
