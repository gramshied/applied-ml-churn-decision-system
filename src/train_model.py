import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline


def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def load_preprocessor():
    return joblib.load("src/preprocessor.joblib")


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(model.__class__.__name__)
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("-" * 50)


def main():
    X_train, X_test, y_train, y_test = load_data()
    preprocessor = load_preprocessor()

    models = [
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    ]

    for model in models:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )
        train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

    # Save best model (we’ll choose after evaluation)
    joblib.dump(pipeline, "src/final_model.joblib")


if __name__ == "__main__":
    main()
