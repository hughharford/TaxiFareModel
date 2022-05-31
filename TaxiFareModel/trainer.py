# imports
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

from TaxiFareModel.pipeline import MakePipeline
from TaxiFareModel.utils import compute_rmse, haversine_vectorized
from TaxiFareModel.data import get_data, clean_data

EXPERIMENT_NAME = "[UK] [London] [hughharford] LinearReg_01"
MLFLOW_URI = "https://mlflow.lewagon.ai/"
class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME




    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_maker = MakePipeline()
        model = LinearRegression() #LassoCV(cv=5, n_alphas=5)

        self.pipeline = pipe_maker.define_full_pipe(model)
        # print(self.pipeline)
        # return self.pipeline

    def run(self):
        """set and train the pipeline"""

        self.set_pipeline()
        self.mlflow_log_param("ModelName", "linear")
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        score = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("RMSE", score)
        return score

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":
    # get data
    data_df = get_data(nrows=10_000)
    # clean data
    data_df_clean = clean_data(data_df)
    # set X and y
    y = data_df_clean["fare_amount"]
    X = data_df_clean.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)
    # hold out
    t = Trainer(X_train, y_train)
    # train
    t.set_pipeline()
    t.run()
    # evaluate
    firstpass_score = t.evaluate(X_test, y_test)

    print(f'done! firstpass_score: {firstpass_score}')
