from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer

class MakePipeline():
    def __init__(self):
        pass

    def create_preproc(self):
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        return preproc_pipe

    def add_model(self, preprocessing_pipe, model=None):

        # Add the model of your choice to the pipeline
        full_pipe = Pipeline([
            ('preproc', preprocessing_pipe),
            ('model', model)
        ])
        return full_pipe

    def define_full_pipe(self, model):
        preproc = self.create_preproc()
        pipeline_with_model = self.add_model(preproc, model)
        return pipeline_with_model
