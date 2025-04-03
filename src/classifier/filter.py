import shelve
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from classifier.filenames import (
    RAW_DATA_FILE,
    FILTER_DATA_FILE,
    FILTER_MODEL_NAME,
    THRESHOLD_KEY,
    SHELVE_FILE,
)


class Filter:
    _model: RandomForestClassifier | None = None

    def __init__(self, df: pd.DataFrame):
        self.X = df.drop("didReOrder", axis=1)
        self.y = df["didReOrder"]

        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

    def create_model(self) -> RandomForestClassifier:
        # Train a lightweight Random Forest as the filtering model
        filter_model = RandomForestClassifier(
            n_estimators=50,  # Fewer trees for speed
            max_depth=10,  # Limit depth for faster prediction
            min_samples_leaf=50,  # Larger leaf size for generalization
            n_jobs=-1,  # Use all cores
            class_weight="balanced",
            random_state=42,
        )

        filter_model.fit(self.X_train, self.y_train)
        return filter_model

    @property
    def model(self) -> RandomForestClassifier:
        if self._model is None:
            self._model = self.create_model()
        return self._model

    def determine_threshold(self) -> np.float64:
        X_test = self.X_test
        y_test = self.y_test

        # Get probabilities for reordering
        reorder_probs = self.model.predict_proba(X_test)[:, 1]

        thresholds = np.linspace(0.01, 0.5, 20)  # Try thresholds from 1% to 50%
        results = []

        for threshold in thresholds:
            # Count customers filtered out (predicted as non-reorders)
            filtered_out = (reorder_probs < threshold).sum()

            # Count false negatives (actual reorders that get filtered out)
            false_negatives = ((reorder_probs < threshold) & (y_test == 1)).sum()

            # Percentage of actual reorders wrongly filtered out
            if y_test.sum() > 0:
                false_negative_rate = false_negatives / y_test.sum()
            else:
                false_negative_rate = 0

            # Percentage of data filtered out
            filter_rate = filtered_out / len(y_test)

            results.append(
                {
                    "threshold": threshold,
                    "filter_rate": filter_rate,
                    "false_negative_rate": false_negative_rate,
                    "filtered_out": filtered_out,
                    "false_negatives": false_negatives,
                }
            )

        results_df = pd.DataFrame(results)

        # Choose threshold where false negative rate is acceptably low (e.g., < 5%)
        # For example:
        optimal_threshold = results_df[results_df["false_negative_rate"] < 0.05][
            "threshold"
        ].max()
        return optimal_threshold


def create_filter_model(filename: str):
    raw_data = pd.read_csv(filename)

    filter_handler = Filter(df=raw_data)
    filter_model = filter_handler.model
    threshold = filter_handler.determine_threshold()

    with shelve.open(SHELVE_FILE) as s:
        s[THRESHOLD_KEY] = threshold

    joblib.dump(value=filter_model, filename=FILTER_MODEL_NAME)


def filter_dataset(
    input_filename: str, threshold: np.float64, output_filename: str
) -> None:
    filter_model = joblib.load(FILTER_MODEL_NAME)

    df = pd.read_csv(input_filename)

    # Get filter model predictions
    filter_probs = filter_model.predict_proba(df.drop("didReOrder", axis=1))[:, 1]

    # Create mask for records that pass the filter
    pass_filter = filter_probs >= threshold

    df[pass_filter].to_csv(output_filename)


def main():
    create_filter_model(filename=RAW_DATA_FILE)

    threshold = np.float64()
    with shelve.open(SHELVE_FILE) as s:
        threshold = s[THRESHOLD_KEY]
    filter_dataset(
        input_filename=RAW_DATA_FILE,
        threshold=threshold,
        output_filename=FILTER_DATA_FILE,
    )


if __name__ == "__main__":
    main()
