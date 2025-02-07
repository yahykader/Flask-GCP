import logging
import os
import pickle

from seaborn import load_dataset
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def load_data() -> tuple:
    """
    Load the iris dataset and return the features and target
    """
    logging.info("📥 Loading dataset")
    df = load_dataset("iris")
    X, y = df.drop("species", axis=1), df["species"]
    y = y.map({"setosa": 0, "versicolor": 1, "virginica": 2})
    logging.info("✅ Dataset loaded")
    return X, y


def train_model(X, y) -> RandomForestClassifier:
    """
    Train a Random Forest model on the iris dataset
    """
    logging.info("🚂 Training model")
    model = RandomForestClassifier()
    model.fit(X, y)
    logging.info(
        "✅ Model trained",
    )
    return model


def save_model(model, cloud: bool = False) -> None:
    pickle.dump(model, open(f"models/model.pkl", "wb"))
    if cloud:
        logging.info("✅ Model saved to cloud")


if __name__ == "__main__":
    print("CLOUD : ", os.environ.get("CLOUD", "PAS DE VARIABLE DEFINIE"))
    X, y = load_data()
    model = train_model(X, y)
    save_model(model, os.environ.get("CLOUD", False))
    logging.info("🚀 Model training pipeline completed")
