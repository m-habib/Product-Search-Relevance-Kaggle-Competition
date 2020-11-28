# This class is responsible for data preprocessing: stemming, spellchecking, etc.
# The final data for training will be available in allForTraining attribute after calling Preprocess function

import pandas as pd
import numpy as np
from pathlib import Path
from src.DataManager import DataManager
from src.FeatureManager import FeatureManager
from src.configuration import config
from src.utils import DfCustomPrintFormat
from nltk.stem.porter import *
from nltk import LancasterStemmer
from sklearn.metrics.pairwise import cosine_similarity

## Dev start

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import sklearn.feature_extraction.text as sktf
from difflib import SequenceMatcher as seq_matcher
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from ngram import NGram
from nltk.stem.porter import *




def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word) >= 0:
            cnt += 1
    return cnt


def ngram_similarity(data, col1, col2):
    cos = []
    for i in range(len(data.id)):
        st = data[col1][i]
        title = data[col2][i]
        n = NGram(title.split(), key=lambda x: x[1])
        for s in st.split():
            n.search(s)

        tfidf = sktf.TfidfVectorizer().fit_transform([st, title])
        c = ((tfidf * tfidf.T).A)[0, 1]
        cos.append(c)
    return cos


def dist_cosine(data, col1, col2):
    cos = []
    for i in range(len(data.id)):
        title = data[col2][i]
        if title is None or len(title) == 0 or title == "nan":
            return 0
        st = data[col1][i]
        tfidf = sktf.TfidfVectorizer().fit_transform([st, title])
        c = ((tfidf * tfidf.T).A)[0, 1]
        cos.append(c)
    return cos


def mean_dist(data, col1, col2):
    mean_edit_s_t = []
    for i in range(len(data)):
        title = data[col2][i]
        if title is None or len(title) == 0 or title == "nan":
            return 0
        search = data[col1][i]
        max_edit_s_t_arr = []
        for s in search.split():
            max_edit_s_t = []
            for t in title.split():
                a = seq_matcher(None, s, t).ratio()
                max_edit_s_t.append(a)
            max_edit_s_t_arr.append(max_edit_s_t)
        l = 0
        for item in max_edit_s_t_arr:
            l = l + max(item)
        mean_edit_s_t.append(l / len(max_edit_s_t_arr))
    return mean_edit_s_t

#serach_term , product title
def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]", " ", str2)
    str2 = [z for z in set(str2.split()) if len(z) > 2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word) > 3:
            s1 = []
            s1 += segmentit(word, str2, True)
            if len(s) > 1:
                s += [z for z in s1 if z not in ['er', 'ing', 's', 'less'] and len(z) > 1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))

#s=word in searchterm, txt_arr = words in prod title
#s = Housing
#txt_arr = [house, black]
def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                # print(s[:-j],s[len(s)-j:])
                s = s[len(s) - j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i == len(st):
            r.append(st[i:])
    return r

#ProdTitle: black door for houses

def str_whole_word(searchTerm, prodTitle, i_):
    cnt = 0
    while i_ < len(prodTitle):
        i_ = prodTitle.find(searchTerm, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(searchTerm)
    return cnt


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_



## Dev env









def CleanData(s):
    strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
    if s is not None and isinstance(s, str) and len(s) > 0:
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
        s = s.replace(")", "")  # Remove irrelevant chars
        s = s.replace("(", "")  # Remove irrelevant chars
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
        return ""


def Stem(s):
    if s is not None and isinstance(s, str) and len(s) > 0:
        stemmer = LancasterStemmer()
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s = s.lower()
        return s
    else:
        return ""


def Lemmatize(s):
    if s is not None and isinstance(s, str) and len(s) > 0:
        s = s.lower()
        s = s.replace("bbq", "barbeque")
        return s
    else:
        return ""


def SpellCorrect(s):
    if s is not None and isinstance(s, str) and len(s) > 0:
        s = s.lower()
        s = s.replace("toliet", "toilet")
        s = s.replace("vynal", "vinyl")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("plexigla", "plexi gla")
        s = s.replace("vinal", "vinyl")
        s = s.replace("snowbl", "snow bl")
        s = s.replace("whirpool", "whirlpool")
        s = s.replace("bbq", "barbeque")
        s = s.replace("rustoleum", "rust-oleum")
        s = s.replace("airconditioner", "air conditioner")
        s = s.replace("whirlpoolstainless", "whirlpool stainless")
        s = s.replace("skill", "skil")
        #TODO: use spell checker
        return s
    else:
        return ""


def CleanAndNormalize(s):
    if isinstance(s, str):
        s = CleanData(s)
        s = SpellCorrect(s)
        s = Stem(s)
        s = Lemmatize(s)
        return s
    else:
        return ""


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
        s = s.replace("toliet", "toilet")
        s = s.replace("vynal", "vinyl")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("plexigla", "plexi gla")
        s = s.replace("vinal", "vinyl")
        s = s.replace("snowbl", "snow bl")
        s = s.replace("whirpool", "whirlpool")
        s = s.replace("bbq", "barbeque")
        s = s.replace("rustoleum", "rust-oleum")
        s = s.replace("airconditioner", "air conditioner")
        s = s.replace("whirlpoolstainless", "whirlpool stainless")
        s = s.replace("skill", "skil")
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
            self.allDf1 = pd.read_csv(config.allCombinedPath.format('1'), na_filter=False)
            self.allDf1 = self.allDf1.drop(self.allDf1.columns[0], axis=1)
        else:
            self.allDf1 = pd.concat((data.trainDf, data.testDf), axis=0, ignore_index=True)
            self.allDf1 = pd.merge(self.allDf1, data.descriptionDf, how='left', on='product_uid')
            self.allDf1 = pd.merge(self.allDf1, features.brandNameDf, how='left', on='product_uid')
            self.allDf1 = pd.merge(self.allDf1, features.colorDf, how='left', on='product_uid')
            self.allDf1 = pd.merge(self.allDf1, features.materialDf, how='left', on='product_uid')
            self.allDf1.to_csv(config.allCombinedPath.format('1'), na_rep='')
        print('   1. allDf1 - All combined: \n\n', DfCustomPrintFormat(self.allDf1.head()))

        # Clean and normalize
        print('   Cleaning and Normalize data...')
        if Path(config.allCombinedPath.format('2')).is_file():
            print('   ' + config.allCombinedPath.format('2') + ' already exists. Loading...')
            self.allDf2 = pd.read_csv(config.allCombinedPath.format('2'), na_filter=False)
            self.allDf2 = self.allDf2.drop(self.allDf2.columns[0], axis=1)
        else:
            # Clean and normalize
            self.allDf2 = self.allDf1
            self.allDf2['search_term'] = self.allDf2['search_term'].map(lambda x: CleanAndNormalize(x))
            self.allDf2['product_title'] = self.allDf2['product_title'].map(lambda x: CleanAndNormalize(x))
            self.allDf2['product_description'] = self.allDf2['product_description'].map(lambda x: CleanAndNormalize(x))
            self.allDf2['brand'] = self.allDf2['brand'].map(lambda x: CleanAndNormalize(x))
            self.allDf2['color'] = self.allDf2['color'].astype(str).map(lambda x: CleanAndNormalize(x))
            self.allDf2['material'] = self.allDf2['material'].astype(str).map(lambda x: CleanAndNormalize(x))
            self.allDf2['product_info'] = self.allDf2['search_term'] + "\t" + self.allDf2['product_title'] + "\t" + self.allDf2['product_description']
            # Count words
            self.allDf2['len_of_query'] = self.allDf2['search_term'].map(lambda x: len(x.split())).astype(np.int64)
            self.allDf2['len_of_title'] = self.allDf2['product_title'].map(lambda x: len(x.split())).astype(np.int64)
            self.allDf2['len_of_description'] = self.allDf2['product_description'].map(lambda x: len(x.split())).astype(np.int64)
            self.allDf2['len_of_brand'] = self.allDf2['brand'].map(lambda x: len(x.split())).astype(np.int64)
            self.allDf2['len_of_color'] = self.allDf2['color'].astype(str).map(lambda x: len(x.split())).astype(np.int64)
            self.allDf2['len_of_material'] = self.allDf2['material'].astype(str).map(lambda x: len(x.split())).astype(np.int64)
            self.allDf2.to_csv(config.allCombinedPath.format('2'), na_rep='')
        print('   2. allDf2 - After cleaning and normalizing: \n\n', DfCustomPrintFormat(self.allDf2.head()))


        if Path(config.allCombinedPath.format('4')).is_file():
            print('   ' + config.allCombinedPath.format('4') + ' already exists. Loading...')
            self.allDf4 = pd.read_csv(config.allCombinedPath.format('4'), na_filter=False)
            self.allDf4 = self.allDf4.drop(self.allDf4.columns[0], axis=1)
        else:
            self.allDf4 = self.allDf2
            self.allDf4['search_term'] = self.allDf4['product_info'].map(lambda x: seg_words(x.split('\t')[0], x.split('\t')[1]))
            self.allDf4['query_in_title'] = self.allDf4['product_info'].map(lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[1], 0))
            self.allDf4['query_in_description'] = self.allDf4['product_info'].map(lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[2], 0))
            self.allDf4['query_last_word_in_title'] = self.allDf4['product_info'].map(lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[1]))
            self.allDf4['query_last_word_in_description'] = self.allDf4['product_info'].map(lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[2]))
            self.allDf4["cosine_s.brand"] = dist_cosine(self.allDf4, "search_term", "brand")
            self.allDf4["cosine_s.material"] = dist_cosine(self.allDf4, "search_term", "material")
            self.allDf4["cosine_s.title"] = dist_cosine(self.allDf4, "search_term", "product_title")
            self.allDf4["mean_s.brand"] = mean_dist(self.allDf4, "search_term", "brand")
            self.allDf4["mean_s.material"] = mean_dist(self.allDf4, "search_term", "material")
            self.allDf4["mean_s.title"] = mean_dist(self.allDf4, "search_term", "product_title")
            self.allDf4.to_csv(config.allCombinedPath.format('4'), na_rep='')
        print('4. allDf4 \n', self.allDf4.head())
