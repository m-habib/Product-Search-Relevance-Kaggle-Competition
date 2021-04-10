# This class is responsible for feature extraction and engineering

import pandas as pd
from pathlib import Path
from src.DataManager import DataManager
from src.configuration import config
from src.utils import DfCustomPrintFormat


class FeatureManager:
    def __init__(self):
        self.brandNameDf = pd.DataFrame()
        self.containsColorDf = pd.DataFrame()
        self.colorDf = pd.DataFrame()
        self.containsMaterialDf = pd.DataFrame()
        self.materialDf = pd.DataFrame()
        self.bagOfWordsDf = pd.DataFrame()

    def EngineerFeatures(self, data):
        print("Features engineering...")

        # Bag Of Words
        print("   Bag Of Words...")
        if Path(config.bagOfWordsPath).is_file():
            print('   ' + config.bagOfWordsPath + ' already exists. Loading...')
            self.bagOfWordsDf = pd.read_csv(config.bagOfWordsPath, na_filter=False, header=0, names=["product_uid", "bag_of_words"])
        else:
            self.bagOfWordsDf = data.trainDf.loc[data.trainDf['relevance'] >= 2][["product_uid", "search_term"]].rename(columns={"search_term": "bag_of_words"}) # Select search terms where relevance is > 2 and join search terms for each prod_uid
            self.bagOfWordsDf['bag_of_words'] = self.bagOfWordsDf['bag_of_words'].astype(str)
            self.bagOfWordsDf = self.bagOfWordsDf.groupby('product_uid', as_index=False).agg({'bag_of_words': lambda x: ' '.join(x)})
            self.bagOfWordsDf.to_csv(config.bagOfWordsPath, na_rep='')
        #print('   bagOfWordsDf: \n   {0}\n'.format(DfCustomPrintFormat(self.bagOfWordsDf.head())))

        # Brand Name
        print("   Brand name...")
        if Path(config.brandNamePath).is_file():
            print('   ' + config.brandNamePath + ' already exists. Loading...')
            self.brandNameDf = pd.read_csv(config.brandNamePath, na_filter=False, header=0, names=["product_uid", "brand"])
        else:
            self.brandNameDf = data.attributesDf[data.attributesDf.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
            self.brandNameDf['brand'] = self.brandNameDf['brand'].astype(str)
            self.brandNameDf.to_csv(config.brandNamePath, na_rep='')
        #print('   brandNameDf: \n   {0}\n'.format(DfCustomPrintFormat(self.brandNameDf.head())))

        # Color
        print("   Color...")
        if Path(config.containsColorPath).is_file():
            print('   ' + config.containsColorPath + ' already exists. Loading...')
            self.containsColorDf = pd.read_csv(config.containsColorPath, na_filter=False, header=0, names=["product_uid", "color"])
        else:
            self.containsColorDf = data.attributesDf[data.attributesDf.name.str.contains("Color", na=False)][["product_uid", "value"]].rename(columns={"value": "color"})
            self.containsColorDf['color'] = self.containsColorDf['color'].astype(str)
            self.containsColorDf.to_csv(config.containsColorPath, na_rep='')
        #print('   containsColorDf: \n   ', DfCustomPrintFormat(self.containsColorDf.head()))
        if Path(config.colorPath).is_file():
            print('   ' + config.colorPath + ' already exists. Loading...')
            self.colorDf = pd.read_csv(config.colorPath, na_filter=False, header=0, names=["product_uid", "color"])
        else:
            self.colorDf = self.containsColorDf.groupby('product_uid', as_index=False).agg(lambda x: ' '.join(x))
            self.colorDf['color'] = self.colorDf['color'].astype(str)
            self.colorDf.to_csv(config.colorPath, na_rep='')
        #print('   colorDf: \n   {0}\n'.format(DfCustomPrintFormat(self.colorDf.head())))

        # Material
        print("   Material...")
        if Path(config.containsMaterialPath).is_file():
            print('   ' + config.containsMaterialPath + ' already exists. Loading...')
            self.containsMaterialDf = pd.read_csv(config.containsMaterialPath, na_filter=False, header=0, names=["product_uid", "material"])
        else:
            self.containsMaterialDf = data.attributesDf[data.attributesDf.name.str.contains("Material", na=False)][["product_uid", "value"]].rename(columns={"value": "material"})
            self.containsMaterialDf['material'] = self.containsMaterialDf['material'].astype(str)
            self.containsMaterialDf.to_csv(config.containsMaterialPath, na_rep='')
        #print('   containsMaterialDf: \n   ', (DfCustomPrintFormat(self.containsMaterialDf.head())))
        if Path(config.materialPath).is_file():
            print('   ' + config.materialPath + ' already exists. Loading...')
            self.materialDf = pd.read_csv(config.materialPath, na_filter=False, header=0, names=["product_uid", "material"])
        else:
            self.materialDf = self.containsMaterialDf.groupby('product_uid', as_index=False).agg(lambda x: ' '.join(x))
            self.materialDf['material'] = self.materialDf['material'].astype(str)
            self.materialDf.to_csv(config.materialPath, na_rep='')
        #print('   materialDf: \n   {0}\n'.format(DfCustomPrintFormat(self.materialDf.head())))

        print("Finished features engineering")