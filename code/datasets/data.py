import os
import zenml
import pandas as pd
import dvc.api
from zenml.client import Client
from sklearn.model_selection import train_test_split


def extract(version=None) -> tuple[pd.DataFrame, str]:
    data_path = "data/raw/data.csv"
    data_store = "myremote"
    if version == None: version = "v2" 
    path = dvc.api.get_url(
        rev=version, path=data_path, remote=data_store, repo="/home/amir/MlOps"
    )
    df = pd.read_csv(path)
    return df, version

def extract_preprocessed(name, version, size=1):
    client = Client()
    l = client.list_artifact_versions(name = name, tag = version, sort_by="version").items
    latest_artifact = sorted(l, key=lambda x: x.created)[-1]
    df = latest_artifact.load()
    df = df.sample(frac = size, random_state = 88)

    print("size of df is ", df.shape)
    print("df columns: ", df.columns)

    return df

def transform(df:pd.DataFrame):
    df=df.dropna()
    df = df.rename(columns={'2urvived': 'survived'})
    
    X = df.drop('survived', axis=1)
    y = df['survived']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    train = pd.concat([X_train,y_train],axis=1) 
    test = pd.concat([X_test,y_test],axis=1)
    return train,test

def save(train:pd.DataFrame, test:pd.DataFrame, version:str):
    train.to_csv(f"/home/amir/MlOps/data/processed/train.csv")    
    test.to_csv(f"/home/amir/MlOps/data/processed/test.csv")

    zenml.save_artifact(data=train, name="train", tags=[version])
    zenml.save_artifact(data=test, name="test", tags=[version])


df,version = extract()
train,test = transform(df)
save(train,test,version)