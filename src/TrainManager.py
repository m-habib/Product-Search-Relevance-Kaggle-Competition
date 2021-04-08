import pandas as pd
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import linear_model
import numpy as np
from src.configuration import config


def CalculateRmse(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5


def Train(data):
    print("Starting Model Training... ")

    # print(np.isnan(data.allForTraining.any()) + "|False")  # and gets False
    # print(np.isfinite(data.allForTraining.all()) + "|True")  # and gets True

    num_train = data.allDf1.shape[0]//2 #TODO: fix
    print("num_train: {0}\n\n".format(num_train))

    trainDf = data.allForTraining.iloc[:num_train]
    testDf = data.allForTraining.iloc[num_train:]
    id_test = testDf['id']

    y_train = trainDf['relevance'].values
    X_train = trainDf.drop(['id', 'relevance'], axis=1).values
    X_test = testDf.drop(['id', 'relevance'], axis=1).values

    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    yo_pred = clf.predict(X_train)

    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(config.linearRegressionOutputPath , index=False)

    RSME = CalculateRmse(y_train, yo_pred)
    print("RSME = ", RSME)
