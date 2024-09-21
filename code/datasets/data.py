import json
import os
import zenml
import pandas as pd
import numpy as np
import dvc.api
from zenml.client import Client
from great_expectations.data_context import FileDataContext
from hydra import compose, initialize
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from zenml import save_artifact
from zenml.integrations.sklearn.materializers.sklearn_materializer import (
    SklearnMaterializer,
)



def extract_data(version=None) -> tuple[pd.DataFrame, str]:
    data_path = "data/raw/data.csv"
    data_store = "myremote"
    if version == None: version = "v1" 

    path = dvc.api.get_url(
        rev=version, path=data_path, remote=data_store, repo="/home/amir/MlOps"
    )

    df = pd.read_csv(path)

    return df, version

df,version = extract_data()
print(df.head(5))