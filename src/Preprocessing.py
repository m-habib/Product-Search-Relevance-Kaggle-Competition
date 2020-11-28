# This class is responsible for data preprocessing: stemming, spellchecking, etc.
# The final data for training will be available in allForTraining attribute after calling Preprocess function

import pandas as pd
from pathlib import Path
from src.DataManager import DataManager
from src.FeatureManager import FeatureManager
from src.configuration import config
from src.utils import DfCustomPrintFormat
from nltk.stem.porter import *
from nltk import LancasterStemmer


def CleanData(s):
    strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
    if s != "null" and isinstance(s, str):
        s = s.lower()  # To lower case
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)  # Restore new line and replace it with single space
        s = re.sub(r'http.*\s', r'', s)  # Remove Http URLs
        s = s.replace('src=', '')  # Remove HTML code
        s = s.replace('alt=', '')  # Remove HTML code
        s = s.replace('/br', '')  # Remove HTML code
        s = s.replace('/centerimg', '')  # Remove HTML code
        s = s.replace('/centerbr', '')  # Remove HTML code
        s = s.replace(':br', '')  # Remove HTML code
        s = s.replace("&nbsp;", " ")  # Replace HTML non breaking space with single space
        s = " ".join(s.split())  # Replace whitespace with single space
        s = re.sub(r"([0-9]),([0-9])", r"\1 \2", s)  # Replace ',' separating numbers with single space
        s = s.replace(",", "")  # Remove ','
        s = s.replace("$", " ")  # Remove irrelevant chars
        s = s.replace("\"", " ")  # Remove irrelevant chars
        s = s.replace("?", " ")  # Remove irrelevant chars
        s = s.replace("-", " ")  # Remove irrelevant chars
        s = s.replace(";", "")  # Remove irrelevant chars
        s = s.replace(":", " ")  # Remove irrelevant chars
        s = s.replace("//", "/")  # Remove duplicates
        s = s.replace("/", " ")  # Remove irrelevant chars
        s = s.replace("\\\\", "\\")  # Remove duplicates
        s = s.replace("\\", " ")  # Remove irrelevant chars
        s = s.replace("..", ".")  # Remove duplicates
        s = s.replace(".", " . ")  # Add space to be able to remove later
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)  # Separate number and char
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)  # Separate number and char
        s = s.replace(" x ", " xby ")  # Replace multiplication or 'by' with 'xby'
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*", " xby ")  # Replace multiplication or 'by' with 'xby'
        s = s.replace(" by ", " xby ")  # Replace multiplication or 'by' with 'xby'
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)  # Separate connected words
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.? ", r"\1in. ", s)  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)  # Uniforms units
        s = s.replace(" v ", " volts ")  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)  # Uniforms units
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)  # Uniforms units
        s = s.replace(" . ", " ")  # Remove irrelevant chars
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])  # Word to number
        s = " ".join(s.split())  # Replace whitespace with single space

        # TODO: bbq
        return s
    else:
        return "null"


def Stem(s):
    if s != "null" and isinstance(s, str):
        stemmer = LancasterStemmer()
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s = s.lower()
        return s
    else:
        return "null"


def Lemmatize(s):
    if s != "null" and isinstance(s, str):
        s = s.lower()
        s = s.replace("bbq", "barbeque")
        return s
    else:
        return "null"


def SpellCorrect(s):
    if s != "null" and isinstance(s, str):
        s = s.lower()
        s = s.replace("bbq", "barbeque")
        s = s.replace("toliet", "toilet")
        s = s.replace("airconditioner", "air conditioner")
        s = s.replace("vinal", "vinyl")
        s = s.replace("vynal", "vinyl")
        s = s.replace("skill", "skil")
        s = s.replace("snowbl", "snow bl")
        s = s.replace("plexigla", "plexi gla")
        s = s.replace("rustoleum", "rust-oleum")
        s = s.replace("whirpool", "whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless", "whirlpool stainless")
        #TODO: use spell checker
        return s
    else:
        return "null"


def CleanAndNormalize(s):
    if isinstance(s, str):
        s = CleanData(s)
        s = SpellCorrect(s)
        s = Stem(s)
        s = Lemmatize(s)
        return s
    else:
        return "null"


class Preprocessor:
    def __init__(self):
        self.allDf1 = pd.DataFrame()
        self.allDf2 = pd.DataFrame()
        self.allDf3 = pd.DataFrame()
        self.allDf4 = pd.DataFrame()
        self.allDf5 = pd.DataFrame()
        self.allForTraining = pd.DataFrame()

    def Preprocess(self, data, features):
        print('Preprocessing...')

        # Combine train and test data with product description and extracted features in one DF
        print('   Combining all...')
        if Path(config.allCombinedPath.format('1')).is_file():
            print('   ' + config.allCombinedPath.format('1') + ' already exists. Loading...')
            self.allDf1 = pd.read_csv(config.allCombinedPath.format('1'))
            self.allDf1 = self.allDf1.drop(self.allDf1.columns[0], axis=1)
        else:
            self.allDf1 = pd.concat((data.trainDf, data.testDf), axis=0, ignore_index=True)
            self.allDf1 = pd.merge(self.allDf1, data.descriptionDf, how='left', on='product_uid')
            self.allDf1 = pd.merge(self.allDf1, features.brandNameDf, how='left', on='product_uid')
            self.allDf1 = pd.merge(self.allDf1, features.colorDf, how='left', on='product_uid')
            self.allDf1 = pd.merge(self.allDf1, features.materialDf, how='left', on='product_uid')
            self.allDf1.to_csv(config.allCombinedPath.format('1'))
        print('   1. allDf1 \n', DfCustomPrintFormat(self.allDf1.head()))

        print('   Cleaning and Normalize data...')
        if Path(config.allCombinedPath.format('2')).is_file():
            print('   ' + config.allCombinedPath.format('2') + ' already exists. Loading...')
            self.allDf2 = pd.read_csv(config.allCombinedPath.format('2'))
            self.allDf2 = self.allDf2.drop(self.allDf2.columns[0], axis=1)
        else:
            self.allDf2 = self.allDf1
            self.allDf2['search_term'] = self.allDf2['search_term'].map(lambda x: CleanAndNormalize(x))
            self.allDf2['product_title'] = self.allDf2['product_title'].map(lambda x: CleanAndNormalize(x))
            self.allDf2['product_description'] = self.allDf2['product_description'].map(lambda x: CleanAndNormalize(x))
            self.allDf2['brand'] = self.allDf2['brand'].map(lambda x: CleanAndNormalize(x))
            self.allDf2['color'] = self.allDf2['color'].astype(str).map(lambda x: CleanAndNormalize(x))
            self.allDf2['material'] = self.allDf2['material'].astype(str).map(lambda x: CleanAndNormalize(x))
            self.allDf2['product_info'] = self.allDf2['search_term'] + "\t" + self.allDf2['product_title'] + "\t" + self.allDf2['product_description']
            self.allDf2.to_csv(config.allCombinedPath.format('2'))
        print('   2. allDf2 \n', DfCustomPrintFormat(self.allDf2.head()))