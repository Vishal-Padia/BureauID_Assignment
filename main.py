import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score


# Load the data
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


# Preprocess the data
def preprocess_data(train_data, test_data):
    # Separate features and target
    X = train_data.drop("Application Status", axis=1)
    y = train_data["Application Status"]

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return X, y, preprocessor, test_data


# Train the model
def train_model(X, y, preprocessor):
    # Create a pipeline
    clf = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # Train the model
    clf.fit(X, y)

    return clf


# Make predictions
def make_predictions(clf, test_data):
    # Prepare test data (remove UID if present)
    test_features = (
        test_data.drop("UID", axis=1) if "UID" in test_data.columns else test_data
    )

    # Predict
    predictions = clf.predict(test_features)

    # Create submission dataframe
    submission = pd.DataFrame({"UID": test_data["UID"], "Prediction": predictions})

    # Save predictions
    submission.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")

    return submission


# Main execution
def main():
    # Paths to the data files
    train_path = "data/Assignment_Train.csv"
    test_path = "data/Assignment_Test.csv"

    # Load data
    train_data, test_data = load_data(train_path, test_path)

    # Preprocess data
    X, y, preprocessor, test_data = preprocess_data(train_data, test_data)

    # Train model
    clf = train_model(X, y, preprocessor)

    # Optional: Evaluate model performance on training data
    train_predictions = clf.predict(X)
    print("Training Accuracy:", accuracy_score(y, train_predictions))
    print("\nClassification Report:\n", classification_report(y, train_predictions))

    # Make predictions on test data
    submissions = make_predictions(clf, test_data)

    return submissions


# Run the script
if __name__ == "__main__":
    main()
