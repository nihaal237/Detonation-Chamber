import argparse

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_SEED = 42

OBSERVATION_WINDOWS = [10, 30, 60, 180]

FEATURE_COLUMNS = [
    "file_size",
    "number_of_sections",
    "number_of_imports",
    "average_section_entropy",
    "process_create_count",
    "file_create_count",
    "registry_set_count"
]


# --------------------------------------------------
# 2. Generate Synthetic Demonstration Data
# --------------------------------------------------

def create_demo_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    records = []

    # Create 100 synthetic executable identities

    for sample_index in range(100):

        label = sample_index % 2

        sample_id = f"synthetic_{sample_index:04d}"

        # Generate example static features

        file_size = int(
            rng.integers(50000, 500000)
        )

        sections = int(
            rng.integers(3, 10)
        )

        imports = int(
            rng.integers(20, 300)
        )

        entropy = float(
            rng.uniform(3.0, 7.5)
        )

        # Each sample has four observation windows

        for window in OBSERVATION_WINDOWS:

            # Artificial patterns used only to
            # test whether training works.

            process_count = int(
                rng.poisson(
                    2 + label * 3
                )
            )

            file_count = int(
                rng.poisson(
                    3 + label * 8
                )
            )

            registry_count = int(
                rng.poisson(
                    1 + label * 4
                )
            )

            records.append({

                "sample_id": sample_id,

                "observation_window": window,

                "file_size": file_size,

                "number_of_sections": sections,

                "number_of_imports": imports,

                "average_section_entropy": entropy,

                "process_create_count": process_count,

                "file_create_count": file_count,

                "registry_set_count": registry_count,

                "label": label
            })

    return pd.DataFrame(records)


# --------------------------------------------------
# 3. Split by Sample Identity
# --------------------------------------------------

def split_dataset(df):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_SEED
    )

    train_indices, test_indices = next(
        splitter.split(
            df,
            df["label"],
            groups=df["sample_id"]
        )
    )

    train_df = df.iloc[train_indices].copy()

    test_df = df.iloc[test_indices].copy()

    # Verify that sample identities do not overlap

    train_samples = set(
        train_df["sample_id"]
    )

    test_samples = set(
        test_df["sample_id"]
    )

    assert train_samples.isdisjoint(test_samples)

    return train_df, test_df


# --------------------------------------------------
# 4. Train Random Forest
# --------------------------------------------------

def train_model(train_df):

    X_train = train_df[FEATURE_COLUMNS]

    y_train = train_df["label"]

    model = RandomForestClassifier(
        n_estimators=100, #100 decision trees
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# --------------------------------------------------
# 5. Evaluate Model
# --------------------------------------------------

def evaluate_model(model, test_df):

    X_test = test_df[FEATURE_COLUMNS]

    y_test = test_df["label"]

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n--------------------------------")
    print("MODEL EVALUATION")
    print("--------------------------------")

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malware"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )
    )


# --------------------------------------------------
# 6. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Train using synthetic demonstration data"
    )

    args = parser.parse_args()

    if not args.demo:

        parser.error(
            "Only --demo is supported currently. "
            "Real-data training will be added after "
            "dataset verification."
        )

    print("\nCreating synthetic demonstration dataset...")

    df = create_demo_dataset()

    print("Total records:", len(df))

    print(
        "Unique samples:",
        df["sample_id"].nunique()
    )

    print("\nLabel distribution:")

    print(
        df.groupby("sample_id")["label"]
        .first()
        .value_counts()
    )

    train_df, test_df = split_dataset(df)

    print("\nTraining records:", len(train_df))

    print("Testing records:", len(test_df))

    print(
        "Training samples:",
        train_df["sample_id"].nunique()
    )

    print(
        "Testing samples:",
        test_df["sample_id"].nunique()
    )

    print("\nTraining Random Forest...")

    model = train_model(train_df)

    print("Training completed!")

    evaluate_model(model, test_df)

    print(
        "\nDEMO ONLY: Results are based on "
        "artificially generated data."
    )

    print(
        "No model was saved to the research artifacts."
    )