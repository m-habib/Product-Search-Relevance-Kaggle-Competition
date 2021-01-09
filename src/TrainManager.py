import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor


def CalculateRmse(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5


def Train(data):
    print("Starting Model Training... ")

    num_train = data.allDf1.shape[0]//2 #TODO: fix
    print("num_train: {0}\n\n".format(num_train))

    trainDf = data.allForTraining.iloc[:num_train]
    testDf = data.allForTraining.iloc[num_train:]
    id_test = testDf['id']

    y_train = trainDf['relevance'].values
    X_train = trainDf.drop(['id', 'relevance'], axis=1).values
    X_test = testDf.drop(['id', 'relevance'], axis=1).values

    # print('X_train \n', X_train.head())
    # print('y_train \n', y_train.head())

    ########################## RandomForestRegressor #################################

    clf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=2016, verbose=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    yo_pred = clf.predict(X_train)

    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../output/submission_V9_RFR.csv', index=False)

    RSME = CalculateRmse(y_train, yo_pred)
    print("RandomForestRegressor - RSME = ", RSME)

    ########################## SVR #################################

    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    yo_pred = clf.predict(X_train)

    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../output/submission_V9_SVR.csv', index=False)

    RSME = CalculateRmse(y_train, yo_pred)
    print("SVR - RSME = ", RSME)

    ########################## LinearRegression #################################

    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    yo_pred = clf.predict(X_train)

    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../output/submission_V9_GLR.csv', index=False)

    RSME = CalculateRmse(y_train, yo_pred)
    print("LinearRegression - RSME = ", RSME)

    ########################## GradientBoostingRegressor #################################

    clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    yo_pred = clf.predict(X_train)

    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../output/submission_V9_GBR.csv', index=False)

    RSME = CalculateRmse(y_train, yo_pred)
    print("GradientBoostingRegressor - RSME = ", RSME)