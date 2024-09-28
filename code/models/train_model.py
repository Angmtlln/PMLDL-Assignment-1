import os
import datetime
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Setuptools is replacing distutils")

BASE_PATH = os.path.expandvars("/home/amir/MlOps/")


def train(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def log_metadata(model, X_test, y_test):
    experiment_name = "classifier_mlops_experiment"
    try: 
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException: 
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id
    print("experiment-id : ", experiment_id)

    if mlflow.active_run(): mlflow.end_run()

    with mlflow.start_run(): pass

    with mlflow.start_run(experiment_id=experiment_id):
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("mean_absolute_error", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.set_tag("model", "basic Random Forest classifier")
        
        model_alias = f"linear_random_forest_classifier_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.sklearn.log_model(model, "model", input_example=X_test)
        print(model_alias)
        
        save_model(model, model_alias)
        


def save_model(model, model_alias):
    mlflow.sklearn.save_model(model, BASE_PATH + "/models/" + model_alias)


def load_model(name):
    return mlflow.sklearn.load_model(BASE_PATH + "/models/" + name)