# Configurations file


class Configuration:
    def __init__(self):
        self.dataPath = "../data"
        self.featuresPath = "../features"
        self.outputPath = "../output"
        self.processStages = self.featuresPath + '/processStages.csv'

        self.brandNamePath = self.featuresPath + '/brandNameDf.csv'
        self.containsColorPath = self.featuresPath + '/containsColorDf.csv'
        self.colorPath = self.featuresPath + '/colorDf.csv'
        self.containsMaterialPath = self.featuresPath + '/containsMaterialDf.csv'
        self.materialPath = self.featuresPath + '/materialDf.csv'
        self.allCombinedPath = self.featuresPath + '/allCombinedDf.csv'


config = Configuration()
