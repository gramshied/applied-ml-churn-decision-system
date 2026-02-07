import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    # Drop identifier column
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges: empty strings -> NaN -> numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing values
    df = df.dropna()

    # Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def split_data(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def save_data(X_train, X_test, y_train, y_test):
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)


def main():
    df = load_data("data/raw.csv")
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)
    save_data(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
