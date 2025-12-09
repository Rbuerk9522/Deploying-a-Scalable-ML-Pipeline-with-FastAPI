import pytest
from ml.model import train_model, inference, compute_model_metrics
from sklearn.linear_model import LogisticRegression
import numpy as np

# TODO: implement the first test. Change the function name and input as needed
def test_model_algorithm_type():
    """
    testing to confirm that the model returns a LogisticRegression model
    """
    
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])

    model = train_model(X, y)

    assert isinstance(model, LogisticRegression)



# TODO: implement the second test. Change the function name and input as needed
def test_inference_output_type_and_shape():
    """
    # testing to guarantee that inference() returns a numpy array and the output length equals the input rows
    """
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)

    pass


# TODO: implement the third test. Change the function name and input as needed
def test_performance_metric_values():
    """
    # testing to check that compute_model_metric() returns the correct types and that all outputs are floats
    """
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])

    model = train_model(X, y)
    preds = inference(model, X)
    p, r, f = compute_model_metrics(y, preds)

    assert isinstance(p, float)
    assert isinstance(r, float)
    assert isinstance(f, float)

    pass
