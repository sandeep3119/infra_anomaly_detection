import mlflow
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def get_mlflow_client(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    return client, experiment_id

def train_model(model_type,X_train, y_train=None,sample_weights=None,params={}):
    model=None
    if model_type == 'xgboost':
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    elif model_type == 'isolation_forest':
        model = IsolationForest(**params)
        model.fit(X_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")    
    
    return model


def evaluate_model(model, X_test, y_test):
    if isinstance(model, IsolationForest):
        y_pred = model.predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert to binary labels
        y_test = [0 if x == 0 else 1 for x in y_test]
    else:
        y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred,zero_division=0),
        "f1_score": classification_report(y_test, y_pred, output_dict=True,zero_division=0)['macro avg']['f1-score']
    }
    return metrics