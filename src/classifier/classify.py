import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from classifier.filenames import FILTER_DATA_FILE


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    df = pd.read_csv(FILTER_DATA_FILE)

    # Step 2: Data preprocessing
    # Handle missing values if any
    df.fillna(
        {
            "orderFrequency": df["orderFrequency"].median(),
            "orderFrequencyStdDev": df["orderFrequencyStdDev"].median(),
        },
        inplace=True,
    )

    # Feature engineering
    df["avgOrderValue"] = df["totalOrderValue"] / df["orderCount"]
    df["itemsPerLine"] = df["lastItemCount"] / df["lastOrderLines"]
    df["daysPerOrder"] = df["lastOrderDaysAgo"] / df["orderCount"]

    # Step 3: Split the data
    X = df.drop("didReOrder", axis=1)
    y = df["didReOrder"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Check class balance
    print("Class distribution:")
    print(y.value_counts(normalize=True))

    # Step 4: Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scale these features
    # Step 5: Build the neural network model
    def build_model(input_dim):
        model = Sequential(
            [
                # Input layer
                Dense(64, input_dim=input_dim, activation="relu"),
                BatchNormalization(),
                Dropout(0.3),
                # Hidden layer
                Dense(32, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                # Output layer
                Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )

        return model

    # Build the model
    model = build_model(X_train_scaled.shape[1])
    model.summary()

    # Step 6: Train the model
    # Set up early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor="val_auc", patience=10, mode="max", restore_best_weights=True
    )

    # Calculate class weights to handle imbalance
    class_counts = y_train.value_counts()
    total = len(y_train)
    class_weight = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}

    # Train the model
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=1,
    )

    # Step 7: Evaluate the model
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["auc"], label="Train AUC")
    plt.plot(history.history["val_auc"], label="Validation AUC")
    plt.title("AUC Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training.pdf")

    # Evaluate on test set
    test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test AUC: {test_results[2]:.4f}")

    # Get predictions
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Print classification metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Step 8: Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
    f1_scores = (
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    )
    optimal_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_idx]

    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"F1-score at optimal threshold: {f1_scores[optimal_threshold_idx]:.4f}")

    # Apply optimal threshold
    y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)
    print("\nClassification Report with Optimal Threshold:")
    print(classification_report(y_test, y_pred_optimal))

    # Step 9: Save the model
    model.save("reorder_prediction_model.keras")
    np.save("feature_scaler.npy", [scaler.mean_, scaler.scale_])

    print("\nModel saved as 'reorder_prediction_model'")
    print("To use this model for predictions:")
    print(
        "1. Load the model: model = tf.keras.models.load_model('reorder_prediction_model')"
    )
    print("2. Load the scaler parameters and create a new scaler")
    print("3. Scale the new data with the loaded scaler")
    print("4. Get predictions: model.predict(X_new_scaled)")
