# This class is responsible for data preprocessing: stemming, spellchecking, etc.
# The final data for training will be available in allForTraining attribute after calling Preprocess function

import pandas as pd
import numpy as np
from pathlib import Path
from src.configuration import config
from src.utils import DfCustomPrintFormat
from nltk import LancasterStemmer
from sklearn.metrics import mean_squared_error
import sklearn.feature_extraction.text as sktf
from nltk.stem.porter import *
from src.configuration import config
from collections import Counter
from gensim.models import Word2Vec
from gensim import models
from scipy import spatial
import math



def CountCommonWords(str1, str2):
    """Counts the common words longer than 2 chars between str1 and str2"""
    str1Words = set(str1.split())
    str2Words = set(str2.split())
    commonWords = str1Words.intersection(str2Words)
    commonWords = [word for word in commonWords if len(word) > 2]
    return len(commonWords)


def CosineSimilarity(data, col1, col2):
    """Calculates Cosine Similarity between corresponding elements in col1 and col2"""
    cos = []
    for i in range(len(data.id)):
        title = data[col2][i]
        if title is None or len(title) == 0 or title == "nan":
            cos.append(0)
            continue
        st = data[col1][i]
        tfidf = sktf.TfidfVectorizer().fit_transform([st, title])
        c = ((tfidf * tfidf.T).A)[0, 1]
        cos.append(c)
    return cos


def CountOccurrences(givenString, substring):
    """Counts occurences of substring in givenString"""
    if len(substring) < 2:
        return 0
    return givenString.count(substring)


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


def CleanAndSpellCorrect(s):
    if isinstance(s, str):
        #print('Log: CleanAndSpellCorrect: ' + str(s))
        s = CleanData(s)
        s = SpellCorrect(s)
        return s
    else:
        return ""


def StemAndLemmatize(s):
    if isinstance(s, str):
        #print('Log: StemAndLemmatize: ' + str(s))
        s = Stem(s)
        s = Lemmatize(s)
        return s
    else:
        return ""


def CalculateAvgSentenceVector(words, model):
    """Calculates the average vector of all words in the given words list"""
    avgVector = np.zeros((300,), dtype="float32")
    wordsCount = 0
    for word in words:
        if word in model:
            wordsCount += 1
            avgVector = np.add(avgVector, model[word])
    if wordsCount > 0:
        avgVector = np.divide(avgVector, wordsCount)
    return avgVector


def word2vecExactMatch(words1, words2, word2vecModel):
    '''Counts the number of words in words1 that doesn't exist in Google's word2vec model but is a common word between words1 and words2'''
    count = 0
    for word in words1:
        if word not in word2vecModel and word in words2:
            #print('     Word not in model: ' + word)
            count += 1
    return count


def Word2VecRelevance(sentence1, sentence2, word2vecModel):
    # print('Log: Word2VecRelevance: ')
    # print('     1. ' + str(sentence1))
    # print('     2. ' + str(sentence2))
    sentence1Words = sentence1.split()
    sentence2Words = sentence2.split()
    similarity = 0
    try:
        sentence1AvgVector = CalculateAvgSentenceVector(sentence1Words, word2vecModel)
        sentence2AvgVector = CalculateAvgSentenceVector(sentence2Words, word2vecModel)
        similarity = 1 - spatial.distance.cosine(sentence1AvgVector, sentence2AvgVector)
    except Exception as e:
        # print('Error Message: ' + str(e))
        # print('keep similarity 0')
        similarity = 0
    # if math.isnan(similarity):
    #     print('replacing nan with 0')
    #     similarity = 0

    exactMatch = word2vecExactMatch(sentence1Words, sentence2Words, word2vecModel)
    if exactMatch > 0:
        print('     exact match: ' + str(exactMatch))

    similarity += exactMatch
    # print('     similarity. ' + str(similarity))
    return similarity


def BagOfWordsRelevance(bag_of_words, search_term):
    # print('Log: BagOfWordsRelevance: ')
    # print('     bagOfWords.  ' + str(bag_of_words))
    # print('     SearchTerm.  ' + str(search_term))
    dictionary = bag_of_words.split()
    dictionary = [word for word in dictionary if len(word) > 2]
    search_term_words_list = search_term.split()
    search_term_words_list = [word for word in search_term_words_list if len(word) > 2]
    if bag_of_words == 0 or len(dictionary) == 0 or len(search_term_words_list) == 0:
        return 0
    common_words = set(dictionary).intersection(search_term_words_list)
    result = len(common_words) / len(search_term_words_list)
    # print('     Result:  ' + str(result))
    return result


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
            self.allDf1 = pd.merge(self.allDf1, features.bagOfWordsDf, how='left', on='product_uid')
            self.allDf1.to_csv(config.allCombinedPath.format('1'), na_rep='')
        self.allDf1.fillna(0, inplace=True)


        # Clean and normalize
        print('   Cleaning and Normalize data...')
        if Path(config.allCombinedPath.format('2')).is_file():
            print('   ' + config.allCombinedPath.format('2') + ' already exists. Loading...')
            self.allDf2 = pd.read_csv(config.allCombinedPath.format('2'), na_filter=False)
            self.allDf2 = self.allDf2.drop(self.allDf2.columns[0], axis=1)
        else:
            # Clean and normalize
            self.allDf2 = self.allDf1
            self.allDf2['search_term'] = self.allDf2['search_term'].map(lambda x: CleanAndSpellCorrect(x))
            self.allDf2['product_title'] = self.allDf2['product_title'].map(lambda x: CleanAndSpellCorrect(x))
            self.allDf2['product_description'] = self.allDf2['product_description'].map(lambda x: CleanAndSpellCorrect(x))
            self.allDf2['brand'] = self.allDf2['brand'].map(lambda x: CleanAndSpellCorrect(x))
            self.allDf2['color'] = self.allDf2['color'].astype(str).map(lambda x: CleanAndSpellCorrect(x))
            self.allDf2['material'] = self.allDf2['material'].astype(str).map(lambda x: CleanAndSpellCorrect(x))
            self.allDf2['bag_of_words'] = self.allDf2['bag_of_words'].astype(str).map(lambda x: CleanAndSpellCorrect(x))

            self.allDf2.fillna(0, inplace=True)
            self.allDf2.to_csv(config.allCombinedPath.format('2'), na_rep='')
        self.allDf2.fillna(0, inplace=True)


        # Feature engineering - Word2Vec
        print('   Feature Engineering 3...')
        if Path(config.allCombinedPath.format('3')).is_file():
            print('   ' + config.allCombinedPath.format('3') + ' already exists. Loading...')
            self.allDf3 = pd.read_csv(config.allCombinedPath.format('3'), na_filter=False)
            self.allDf3 = self.allDf3.drop(self.allDf3.columns[0], axis=1)
        else:
            self.allDf3 = self.allDf2
            # Word2Vec
            print('      Word2Vec relevance...')
            print('      Loading GoogleNews-vectors-negative300.bin...')
            word2vecModel = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
            print('      Loaded successfully')
            self.allDf3['word2vec_term_title'] = self.allDf3.apply(lambda x: Word2VecRelevance(x['search_term'], x['product_title'], word2vecModel), axis=1)  # word2vec similarity between search term and product title
            self.allDf3['word2vec_term_descr'] = self.allDf3.apply(lambda x: Word2VecRelevance(x['search_term'], x['product_description'], word2vecModel), axis=1)  # word2vec similarity between search term and product description
            self.allDf3['word2vec_term_color'] = self.allDf3.apply(lambda x: Word2VecRelevance(x['search_term'], x['color'], word2vecModel), axis=1)  # word2vec similarity between search term and product color
            self.allDf3['word2vec_term_mater'] = self.allDf3.apply(lambda x: Word2VecRelevance(x['search_term'], x['material'], word2vecModel), axis=1)  # word2vec similarity between search term and product material
            self.allDf3['word2vec_term_brand'] = self.allDf3.apply(lambda x: Word2VecRelevance(x['search_term'], x['brand'], word2vecModel), axis=1)  # word2vec similarity between search term and product brand
            self.allDf3.fillna(0, inplace=True)
            self.allDf3.to_csv(config.allCombinedPath.format('3'), na_rep='')
        self.allDf3.fillna(0, inplace=True)


        # Feature engineering - Bag of Words
        print('   Feature Engineering 4...')
        if Path(config.allCombinedPath.format('4')).is_file():
            print('   ' + config.allCombinedPath.format('4') + ' already exists. Loading...')
            self.allDf4 = pd.read_csv(config.allCombinedPath.format('4'), na_filter=False)
            self.allDf4 = self.allDf4.drop(self.allDf4.columns[0], axis=1)
        else:
            self.allDf4 = self.allDf3

            # Stem and Lemmatize
            self.allDf4['search_term'] = self.allDf4['search_term'].map(lambda x: StemAndLemmatize(x))
            self.allDf4['product_title'] = self.allDf4['product_title'].map(lambda x: StemAndLemmatize(x))
            self.allDf4['product_description'] = self.allDf4['product_description'].map(lambda x: StemAndLemmatize(x))
            self.allDf4['brand'] = self.allDf4['brand'].map(lambda x: StemAndLemmatize(x))
            self.allDf4['color'] = self.allDf4['color'].astype(str).map(lambda x: StemAndLemmatize(x))
            self.allDf4['material'] = self.allDf4['material'].astype(str).map(lambda x: StemAndLemmatize(x))
            self.allDf4['bag_of_words'] = self.allDf4['bag_of_words'].astype(str).map(lambda x: StemAndLemmatize(x))

            # Bag of Words relevance
            print('      Bag Of Words relevance...')
            self.allDf4['bag_of_words_relevance'] = self.allDf4.apply(lambda x: BagOfWordsRelevance(x['bag_of_words'], x['search_term']), axis=1)
            self.allDf4.fillna(0, inplace=True)
            self.allDf4.to_csv(config.allCombinedPath.format('4'), na_rep='')
        self.allDf4.fillna(0, inplace=True)


        # Feature engineering - Cosine similarity and Common Words
        print('   Feature Engineering 5...')
        if Path(config.allCombinedPath.format('5')).is_file():
            print('   ' + config.allCombinedPath.format('5') + ' already exists. Loading...')
            self.allDf5 = pd.read_csv(config.allCombinedPath.format('5'), na_filter=False)
            self.allDf5 = self.allDf5.drop(self.allDf5.columns[0], axis=1)
        else:
            self.allDf5 = self.allDf4

            # Count words
            self.allDf5['len_of_query'] = self.allDf5['search_term'].map(lambda x: len(x.split())).astype(np.int64)
            self.allDf5['len_of_brand'] = self.allDf5['brand'].map(lambda x: len(x.split())).astype(np.int64)
            self.allDf5['len_of_color'] = self.allDf5['color'].astype(str).map(lambda x: len(x.split())).astype(np.int64)
            self.allDf5['len_of_material'] = self.allDf5['material'].astype(str).map(lambda x: len(x.split())).astype(np.int64)

            # Count Occurrences of search term as one string in product title and description
            self.allDf5['whole_query_in_title'] = self.allDf5.apply(lambda x: CountOccurrences(x['search_term'], x['product_title']), axis=1)

            # Cosine similarity between search term product title, brand and material
            print('      Cosine Similarity...')
            self.allDf5["title_query_cos"] = CosineSimilarity(self.allDf5, "search_term", "product_title")
            self.allDf5["brand_query_cos"] = CosineSimilarity(self.allDf5, "search_term", "brand")
            # self.allDf5["material_query_cos"] = CosineSimilarity(self.allDf5, "search_term", "material")

            # Common words
            print('      Common Words...')
            self.allDf5['title_query_common_words'] = self.allDf5.apply(lambda x: CountCommonWords(x['search_term'], x['product_title']), axis=1)
            self.allDf5['description_query_common_words'] = self.allDf5.apply(lambda x: CountCommonWords(x['search_term'], x['product_description']), axis=1)
            self.allDf5['brand_query_common_words'] = self.allDf5.apply(lambda x: CountCommonWords(x['search_term'], x['brand']), axis=1)
            self.allDf5['color_query_common_words'] = self.allDf5.apply(lambda x: CountCommonWords(x['search_term'], x['color']), axis=1)
            self.allDf5['material_query_common_words'] = self.allDf5.apply(lambda x: CountCommonWords(x['search_term'], x['material']), axis=1)

            # Common words ratio
            print('      Common Words Ratio...')
            self.allDf5['ratio_title'] = self.allDf5['title_query_common_words'] / self.allDf5['len_of_query']
            self.allDf5['ratio_description'] = self.allDf5['description_query_common_words'] / self.allDf5['len_of_query']
            self.allDf5['ratio_brand'] = self.allDf5['brand_query_common_words'] / self.allDf5['len_of_brand']
            self.allDf5['ratio_color'] = self.allDf5['color_query_common_words'] / self.allDf5['len_of_color']
            self.allDf5['ratio_material'] = self.allDf5['material_query_common_words'] / self.allDf5['len_of_material']

            self.allDf5.fillna(0, inplace=True)
            self.allDf5.to_csv(config.allCombinedPath.format('5'), na_rep='')
        self.allDf5.fillna(0, inplace=True)


        # Prepare for training
        print('   Preparing for training...')
        if Path(config.allForTraining).is_file():
            print('   ' + config.allForTraining + ' already exists. Loading...')
            self.allForTraining = pd.read_csv(config.allForTraining)
            self.allForTraining = self.allForTraining.drop(self.allForTraining.columns[0], axis=1)
        else:
            self.allForTraining = self.allDf5.drop(['search_term', 'product_title', 'product_description', 'brand', 'color', 'material', 'bag_of_words',
                                                    'len_of_query', 'len_of_material', 'len_of_color', 'len_of_brand'], axis=1)
            self.allForTraining.to_csv(config.allForTraining)
        self.allForTraining.fillna(0, inplace=True)