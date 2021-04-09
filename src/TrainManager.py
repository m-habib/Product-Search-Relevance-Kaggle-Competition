import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn import linear_model

from src.configuration import config


def CalculateRmse(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5


def CalculateAbsError(ground_truth, predictions):
    return mean_absolute_error(ground_truth, predictions)


def CalculateAbsolutePercentageError(ground_truth, predictions):
    return mean_absolute_percentage_error(ground_truth, predictions)


def Train(preprocessor, data):
    print("Starting Model Training... ")

    num_train = data.trainDf.shape[0]
    print("Train data size: " + str(num_train))

    trainDf = preprocessor.allForTraining.iloc[:num_train]
    testDf = preprocessor.allForTraining.iloc[num_train:]
    id_test = testDf['id']

    y_train = trainDf['relevance'].values
    X_train = trainDf.drop(['id', 'relevance'], axis=1).values
    X_test = testDf.drop(['id', 'relevance'], axis=1).values

    # RandomForestRegressor
    clf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=2016, verbose=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    yo_pred = clf.predict(X_train)

    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(config.randomForestRegressorOutputPath, index=False)
    print("RandomForestRegressor - RMSE = ", CalculateRmse(y_train, yo_pred))
    print("RandomForestRegressor - ABS = ", CalculateAbsError(y_train, yo_pred))
    print("RandomForestRegressor - ABS Perc = ", CalculateAbsolutePercentageError(y_train, yo_pred))

    # GradientBoostingRegressor
    clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    yo_pred = clf.predict(X_train)
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(config.gradientBoostingRegressorOutputPath, index=False)
    print("GradientBoostingRegressor - RMSE = ", CalculateRmse(y_train, yo_pred))
    print("GradientBoostingRegressor - ABS = ", CalculateAbsError(y_train, yo_pred))
    print("GradientBoostingRegressor - ABS Perc = ", CalculateAbsolutePercentageError(y_train, yo_pred))

    # LinearRegression
    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    yo_pred = clf.predict(X_train)
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(config.linearRegressionOutputPath, index=False)
    print("LinearRegression - RMSE = ", CalculateRmse(y_train, yo_pred))
    print("LinearRegression - ABS = ", CalculateAbsError(y_train, yo_pred))
    print("LinearRegression - ABS Perc = ", CalculateAbsolutePercentageError(y_train, yo_pred))