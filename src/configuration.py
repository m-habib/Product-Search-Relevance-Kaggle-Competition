# Configurations file


class Configuration:
    def __init__(self):
        self.dataPath = "../data"
        self.featuresPath = "../features"
        self.outputPath = "../output"
        self.processStages = '../processStages'

        self.brandNamePath = self.featuresPath + '/brandNameDf.csv'
        self.bagOfWordsPath = self.featuresPath + '/BagOfWords.csv'
        self.containsColorPath = self.featuresPath + '/containsColorDf.csv'
        self.colorPath = self.featuresPath + '/colorDf.csv'
        self.containsMaterialPath = self.featuresPath + '/containsMaterialDf.csv'
        self.materialPath = self.featuresPath + '/materialDf.csv'
        self.allCombinedPath = self.processStages + '/allCombinedDf{0}.csv'
        self.allForTraining = self.processStages + '/allForTraining.csv'
        self.linearRegressionOutputPath = self.outputPath + '/linearRegressionOutput.csv'
        self.gradientBoostingRegressorOutputPath = self.outputPath + '/gradientBoostingRegressorOutput.csv'
        self.randomForestRegressorOutputPath = self.outputPath + '/randomForestRegressorOutput.csv'

        self.bagOfWords = True
        self.tfIdf = True

        self.developmentMode = False
        self.devNumRows = 200


config = Configuration()
