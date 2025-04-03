"""
Use a basic, fast model to estimate feature importance.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from classifier.filenames import FILTER_DATA_FILE


def feature_importance(X, X_train_scaled, y_train) -> pd.DataFrame:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Get feature importance from Random Forest
    importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=False)
    return importance


def main():
    df = pd.read_csv(FILTER_DATA_FILE)

    X = df.drop("didReOrder", axis=1)
    y = df["didReOrder"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    feat_importance = feature_importance(X, X_train_scaled, y_train)

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feat_importance["Feature"], feat_importance["Importance"])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.pdf")


if __name__ == "__main__":
    main()
