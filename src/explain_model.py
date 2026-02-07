import pandas as pd
import joblib
import shap


def load_data():
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").values.ravel()
    return X_test, y_test


def load_pipeline():
    return joblib.load("src/final_model.joblib")


def main():
    X_test, y_test = load_data()
    pipeline = load_pipeline()

    # Extract components
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Transform data
    X_test_transformed = preprocessor.transform(X_test)

    # SHAP explainer
    explainer = shap.Explainer(model, X_test_transformed)
    shap_values = explainer(
    X_test_transformed,
    check_additivity=False
)

    # Global explanation
    shap.summary_plot(shap_values, X_test_transformed, show=False)

    print("SHAP explanation generated successfully.")


if __name__ == "__main__":
    main()
