# imports
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import train_test_split

from TaxiFareModel.pipeline import MakePipeline
from TaxiFareModel.utils import compute_rmse, haversine_vectorized
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

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
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        score = compute_rmse(y_pred, y_test)
        return score


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
