import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")
    y_test = pd.read_csv("data/y_test.csv")
    return X_train, X_test, y_train, y_test


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(exclude="object").columns

    categorical_pipeline = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    numerical_pipeline = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numerical_pipeline, numerical_cols)
        ]
    )

    return preprocessor


def main():
    X_train, X_test, y_train, y_test = load_data()

    preprocessor = build_preprocessor(X_train)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    joblib.dump(preprocessor, "src/preprocessor.joblib")

    print("Feature engineering completed.")


if __name__ == "__main__":
    main()
