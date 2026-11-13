import argparse
import json

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_SEED = 42

ARTIFACT_DIR = Path("artifacts/models")

REPORT_DIR = Path("reports/ml")


STATIC_FEATURES = [
    "file_size",
    "machine_type",
    "number_of_sections",
    "number_of_imports",
    "entry_point",
    "average_section_entropy",
    "maximum_section_entropy",
    "minimum_section_entropy",
    "high_entropy_sections",
    "registry_api_count",
    "network_api_count",
    "process_api_count",
    "file_api_count",
    "imports_registry_api",
    "imports_network_api",
    "imports_process_api",
    "imports_file_api"
]


DYNAMIC_FEATURES = [
    "process_create_count",
    "file_create_count",
    "file_modify_count",
    "file_delete_count",
    "registry_set_count",
    "registry_delete_count",
    "network_attempt_count"
]


FEATURE_GROUPS = {
    "static": STATIC_FEATURES,
    "dynamic": DYNAMIC_FEATURES,
    "combined": STATIC_FEATURES + DYNAMIC_FEATURES
}


# --------------------------------------------------
# 2. Create ML Model
# --------------------------------------------------

def create_model(algorithm):

    if algorithm == "random_forest":

        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1
        )

    elif algorithm == "gradient_boosting":

        classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_SEED
        )

    else:

        raise ValueError(
            f"Unknown algorithm: {algorithm}"
        )

    # The imputer learns replacement values
    # from training data only.

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
                add_indicator=True
            )
        ),
        (
            "classifier",
            classifier
        )
    ])

    return pipeline


# --------------------------------------------------
# 3. Load and Validate Real Dataset
# --------------------------------------------------

def load_real_dataset(
    file_path,
    feature_columns,
    observation_window
):

    df = pd.read_csv(file_path)

    required_columns = [
        "sha256",
        "label",
        "verification_status"
    ] + feature_columns

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Require independently verified labels

    if not (
        df["verification_status"] == "verified"
    ).all():

        raise ValueError(
            "Dataset contains samples "
            "with unverified labels."
        )

    # Keep one observation window

    if "observation_window" in df.columns:

        df = df[
            df["observation_window"] == observation_window
        ].copy()

    if df.empty:

        raise ValueError(
            "No records available for this window."
        )

    # Normalize hashes

    df["sha256"] = (
        df["sha256"]
        .astype(str)
        .str.lower()
    )

    valid_hashes = df["sha256"].str.fullmatch(
        r"[0-9a-f]{64}"
    )

    if not valid_hashes.all():

        raise ValueError(
            "Invalid SHA-256 values found."
        )

    if df["sha256"].duplicated().any():

        raise ValueError(
            "Duplicate sample hashes found."
        )

    # Require binary labels

    if not df["label"].isin([0, 1]).all():

        raise ValueError(
            "Labels must be 0 or 1."
        )

    if df["label"].nunique() != 2:

        raise ValueError(
            "Both benign and malware classes are required."
        )

    # Convert selected features to numeric values

    for column in feature_columns:

        original = df[column]

        converted = pd.to_numeric(
            original,
            errors="coerce"
        )

        invalid = (
            original.notna()
            & converted.isna()
        )

        if invalid.any():

            raise ValueError(
                f"Non-numeric values in {column}."
            )

        df[column] = converted

    return df


# --------------------------------------------------
# 4. Generate Safe Demonstration Dataset
# --------------------------------------------------

def create_demo_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    for index in range(200):

        label = index % 2

        row = {

            "sha256": f"{index:064x}",

            "label": label,

            "verification_status": "synthetic",

            "observation_window": 180
        }

        # Synthetic static features

        for feature in STATIC_FEATURES:

            row[feature] = float(
                rng.uniform(1, 100)
            )

        # Synthetic dynamic features

        for feature in DYNAMIC_FEATURES:

            row[feature] = int(
                rng.poisson(
                    2 + label * 5
                )
            )

        # Demonstrate missing monitoring values

        if index % 10 == 0:

            row["network_attempt_count"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------
# 5. Split Dataset
# --------------------------------------------------

def split_dataset(df):

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )

    # Ensure no SHA-256 overlap

    train_hashes = set(train_df["sha256"])

    test_hashes = set(test_df["sha256"])

    if not train_hashes.isdisjoint(test_hashes):

        raise ValueError(
            "Train/test sample leakage detected."
        )

    return train_df, test_df


# --------------------------------------------------
# 6. Train and Evaluate
# --------------------------------------------------

def train_and_evaluate(
    train_df,
    test_df,
    feature_columns,
    algorithm
):

    model = create_model(algorithm)

    X_train = train_df[feature_columns]

    y_train = train_df["label"]

    X_test = test_df[feature_columns]

    y_test = test_df["label"]

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {

        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "false_positive_rate": float(
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        ),

        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)]
        ],

        "train_samples": len(train_df),

        "test_samples": len(test_df)
    }

    return model, metrics


# --------------------------------------------------
# 7. Save Model and Metrics
# --------------------------------------------------

def save_outputs(
    model,
    metrics,
    algorithm,
    feature_group,
    observation_window
):

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    name = (
        f"{algorithm}_{feature_group}_"
        f"{observation_window}s"
    )

    model_path = ARTIFACT_DIR / f"{name}.joblib"

    metrics_path = REPORT_DIR / f"{name}_metrics.json"

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_GROUPS[feature_group],
            "observation_window": observation_window
        },
        model_path
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\nModel saved to:", model_path)

    print("Metrics saved to:", metrics_path)


# --------------------------------------------------
# 8. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train malware detector on labeled features"
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Verified labeled CSV dataset"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic test data"
    )

    parser.add_argument(
        "--algorithm",
        choices=[
            "random_forest",
            "gradient_boosting"
        ],
        default="random_forest"
    )

    parser.add_argument(
        "--features",
        choices=[
            "static",
            "dynamic",
            "combined"
        ],
        default="combined"
    )

    parser.add_argument(
        "--window",
        type=int,
        choices=[10, 30, 60, 180],
        default=180
    )

    args = parser.parse_args()

    if args.demo == bool(args.input):

        parser.error(
            "Provide exactly one of --demo or --input."
        )

    feature_columns = FEATURE_GROUPS[args.features]

    if args.demo:

        print("\nUsing synthetic demonstration data.")

        df = create_demo_dataset()

    else:

        print("\nLoading verified dataset...")

        df = load_real_dataset(
            args.input,
            feature_columns,
            args.window
        )

    print("Samples:", len(df))

    train_df, test_df = split_dataset(df)

    print("Training samples:", len(train_df))

    print("Testing samples:", len(test_df))

    print(
        f"\nTraining {args.algorithm} "
        f"with {args.features} features..."
    )

    model, metrics = train_and_evaluate(
        train_df,
        test_df,
        feature_columns,
        args.algorithm
    )

    print("\n--------------------------------")
    print("EVALUATION RESULTS")
    print("--------------------------------")

    for key, value in metrics.items():

        print(f"{key}: {value}")

    if args.demo:

        print(
            "\nDEMO ONLY: Synthetic training results."
        )

        print(
            "Model and metrics were not saved."
        )

    else:

        save_outputs(
            model,
            metrics,
            args.algorithm,
            args.features,
            args.window
        )