import pandas as pd
from src.data_preprocessing import data_processing

def test_preprocessing_pipeline_runs():
    df = pd.DataFrame({
        "age": [63, 67, 67],
        "sex": [1, 1, 1],
        "cp": [1, 4, 4],
        "trestbps": [145, 160, 120],
        "chol": [233, 286, 229],
        "fbs": [1, 0, 0],
        "restecg": [2, 2, 2],
        "thalach": [150, 108, 129],
        "exang": [0, 1, 1],
        "oldpeak": [2.3, 1.5, 2.6],
        "slope": [3, 2, 2],
        "ca": [0, 3, 2],
        "thal": [6, 3, 7],
        "target": [0, 1, 1]
    })

    preprocessor, X_train, X_test, y_train, y_test = data_processing(df)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) + len(y_test) == len(df)



