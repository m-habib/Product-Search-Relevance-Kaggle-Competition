# This class is responsible for loading the data into Pandas DF

import pandas as pd
from src.configuration import config


class DataManager:
    def __init__(self):
        self.trainDf = pd.DataFrame()
        self.testDf = pd.DataFrame()
        self.attributesDf = pd.DataFrame()
        self.descriptionDf = pd.DataFrame()

    def LoadData(self):
        print("Loading data...")
        print("   Loading train data...")
        self.trainDf = pd.read_csv(config.dataPath+'/train.csv', encoding="ISO-8859-1")
        print("   Loading test data...")
        self.testDf = pd.read_csv(config.dataPath+'/test.csv', encoding="ISO-8859-1")
        print("   Loading attributes data...")
        self.attributesDf = pd.read_csv(config.dataPath+'/attributes.csv', encoding="ISO-8859-1")
        print("   Loading product description data...")
        self.descriptionDf = pd.read_csv(config.dataPath+'/product_descriptions.csv', encoding="ISO-8859-1")

        if config.developmentMode:
            print("In development mode, taking first {0} rows of data\n".format(config.devNumRows))
            self.trainDf = self.trainDf.head(config.devNumRows)
            self.testDf = self.testDf.head(config.devNumRows)

        print("   Finished loading data\n")


