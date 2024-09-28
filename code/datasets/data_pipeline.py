from hydra import compose, initialize
import pandas as pd
from typing_extensions import Tuple, Annotated
from zenml import step, pipeline, ArtifactConfig
import data


@step(enable_cache=False)
def extract() -> Tuple[
    Annotated[
        pd.DataFrame,
        ArtifactConfig(name="extracted_data", tags=["data_preparation"]),
    ],
    Annotated[str, ArtifactConfig(name="data_version", tags=["data_preparation"])],
]:
    version = 'v2'
    df, version = data.extract()
    print(df.shape, version)
    return df, version


@step(enable_cache=False)
def transform(df: pd.DataFrame) -> Tuple[
    Annotated[
        pd.DataFrame, ArtifactConfig(name="input_features", tags=["data_preparation"])
    ],
    Annotated[
        pd.DataFrame, ArtifactConfig(name="input_target", tags=["data_preparation"])
    ],
]:
    X, y = data.transform(df=df)
    print("Data transformed successfully!")
    return X, y


@step(enable_cache=False)
def save(X: pd.DataFrame, y: pd.DataFrame, version: str):
    data.save(X, y, version)


@pipeline()
def prepare_data_pipeline():
    df,version = extract()
    train,test = transform(df)
    save(train,test,version)
    
    

if __name__ == "__main__":
    run = prepare_data_pipeline()
