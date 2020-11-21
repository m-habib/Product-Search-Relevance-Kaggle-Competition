import pandas as pd
from pathlib import Path
from src.configuration import config
from src.utils import DfCustomPrintFormat

print("Loading data...")
print("   Loading train data...")
trainDf = pd.read_csv(config.dataPath+'/train.csv', encoding="ISO-8859-1")
print("   Loading test data...")
testDf = pd.read_csv(config.dataPath+'/test.csv', encoding="ISO-8859-1")
print("   Loading attributes data...")
attributesDf = pd.read_csv(config.dataPath+'/attributes.csv', encoding="ISO-8859-1")
print("   Loading product description data...")
descriptionDf = pd.read_csv(config.dataPath+'/product_descriptions.csv', encoding="ISO-8859-1")

#Feature Engineering
print("Features Engineering... ")

#Brand Name
print("   Brand name...")
if Path(config.brandNamePath).is_file():
    print('   ' + config.brandNamePath + ' already exists. Loading...')
    brandNameDf = pd.read_csv(config.brandNamePath, header=0, names=["product_uid", "brand"])
else:
    brandNameDf = attributesDf[attributesDf.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
    brandNameDf['brand'] = brandNameDf['brand'].astype(str)
    brandNameDf.to_csv(config.brandNamePath)
print('   brandNameDf: \n   ', DfCustomPrintFormat(brandNameDf.head()))

#Color
print("   Color...")
if Path(config.containsColorPath).is_file():
    print('   ' + config.containsColorPath + ' already exists. Loading...')
    containsColorDf = pd.read_csv(config.containsColorPath, header=0, names=["product_uid", "color"])
else:
    containsColorDf = attributesDf[attributesDf.name.str.contains("Color", na=False)][["product_uid", "value"]].rename(columns={"value": "color"})
    containsColorDf['color'] = containsColorDf['color'].astype(str)
    containsColorDf.to_csv(config.containsColorPath)
print('   containsColorDf: \n   ', DfCustomPrintFormat(containsColorDf.head()))
if Path(config.colorPath).is_file():
    print('   ' + config.colorPath + ' already exists. Loading...')
    colorDf = pd.read_csv(config.colorPath, header=0, names=["product_uid", "color"])
else:
    colorDf = containsColorDf.groupby('product_uid', as_index=False).agg(lambda x: ' '.join(x))
    colorDf['color'] = colorDf['color'].astype(str)
    colorDf.to_csv(config.colorPath)
print('   colorDf: \n   ', DfCustomPrintFormat(colorDf.head()))

#Material
print("   Material...")
if Path(config.containsMaterialPath).is_file():
    print('   ' + config.containsMaterialPath + ' already exists. Loading...')
    containsMaterialDf = pd.read_csv(config.containsMaterialPath, header=0, names=["product_uid", "material"])
else:
    containsMaterialDf = attributesDf[attributesDf.name.str.contains("Material", na=False)][["product_uid", "value"]].rename(columns={"value": "material"})
    containsMaterialDf['material'] = containsMaterialDf['material'].astype(str)
    containsMaterialDf.to_csv(config.containsMaterialPath)
print('   containsMaterialDf: \n   ', DfCustomPrintFormat(containsMaterialDf.head()))
if Path(config.materialPath).is_file():
    print('   ' + config.materialPath + ' already exists. Loading...')
    materialDf = pd.read_csv(config.materialPath, header=0, names=["product_uid", "material"])
else:
    materialDf = containsMaterialDf.groupby('product_uid', as_index=False).agg(lambda x: ' '.join(x))
    materialDf['material'] = materialDf['material'].astype(str)
    materialDf.to_csv(config.materialPath)
print('   materialDf: \n   ', DfCustomPrintFormat(materialDf.head()))

