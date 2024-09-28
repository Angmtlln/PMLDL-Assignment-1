from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats
import pandas as pd


def preprocess(train, test):
    columns_to_drop = [f'zero.{i}' for i in range(1, 19)]
    train = train.drop(columns=columns_to_drop)
    test = test.drop(columns=columns_to_drop)

    train = train.drop(columns=['zero'])
    test = test.drop(columns=['zero'])

    X_train, y_train = train.drop('survived', axis=1), train['survived']
    X_test, y_test = test.drop('survived', axis=1), test['survived']

    return X_train, y_train, X_test, y_test
