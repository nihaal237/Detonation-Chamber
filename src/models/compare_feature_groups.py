import argparse

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from train_random_forest import (
    create_demo_dataset,
    split_dataset,
    RANDOM_SEED
)


# --------------------------------------------------
# 1. Define Feature Groups
# --------------------------------------------------

STATIC_FEATURES = [
    "file_size",
    "number_of_sections",
    "number_of_imports",
    "average_section_entropy"
]

DYNAMIC_FEATURES = [
    "process_create_count",
    "file_create_count",
    "registry_set_count"
]

COMBINED_FEATURES = (
    STATIC_FEATURES + DYNAMIC_FEATURES
)

FEATURE_GROUPS = {
    "Static": STATIC_FEATURES,
    "Dynamic": DYNAMIC_FEATURES,
    "Combined": COMBINED_FEATURES
}


# --------------------------------------------------
# 2. Train One Random Forest Model
# --------------------------------------------------

def train_classifier(train_df, feature_columns):

    X_train = train_df[feature_columns]

    y_train = train_df["label"]

    model = RandomForestClassifier(
        n_estimators=100,
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
# 3. Evaluate One Model
# --------------------------------------------------

def evaluate_classifier(
    model,
    test_df,
    feature_columns
):

    X_test = test_df[feature_columns]

    y_test = test_df["label"]

    predictions = model.predict(X_test)

    return {

        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )
    }


# --------------------------------------------------
# 4. Compare Feature Groups
# --------------------------------------------------

def compare_feature_groups(train_df, test_df):

    results = []

    for group_name, feature_columns in FEATURE_GROUPS.items():

        print(
            f"\nTraining {group_name} model..."
        )

        model = train_classifier(
            train_df,
            feature_columns
        )

        metrics = evaluate_classifier(
            model,
            test_df,
            feature_columns
        )

        results.append({

            "feature_group": group_name,

            "accuracy": metrics["accuracy"],

            "precision": metrics["precision"],

            "recall": metrics["recall"],

            "macro_f1": metrics["macro_f1"]
        })

        print("Training completed.")

        print(
            "Accuracy:",
            round(metrics["accuracy"], 4)
        )

        print(
            "Macro-F1:",
            round(metrics["macro_f1"], 4)
        )

        print("Confusion Matrix:")

        print(metrics["confusion_matrix"])

    return pd.DataFrame(results)


# --------------------------------------------------
# 5. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare Random Forest feature groups"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic demonstration data"
    )

    args = parser.parse_args()

    if not args.demo:

        parser.error(
            "Only --demo is currently supported."
        )

    print("\nCreating synthetic demonstration dataset...")

    df = create_demo_dataset()

    print("Total records:", len(df))

    print(
        "Unique samples:",
        df["sample_id"].nunique()
    )

    # ----------------------------------------------
    # Use the Full Observation Window
    # ----------------------------------------------

    df = df[
        df["observation_window"] == 180
    ].copy()

    print("\nUsing observation window: 180 seconds")

    print(
        "Records selected:",
        len(df)
    )

    # ----------------------------------------------
    # Create One Shared Train/Test Split
    # ----------------------------------------------

    train_df, test_df = split_dataset(df)

    print(
        "\nTraining samples:",
        train_df["sample_id"].nunique()
    )

    print(
        "Testing samples:",
        test_df["sample_id"].nunique()
    )

    # ----------------------------------------------
    # Compare Models
    # ----------------------------------------------

    results = compare_feature_groups(
        train_df,
        test_df
    )

    print("\n--------------------------------")
    print("FEATURE GROUP COMPARISON")
    print("--------------------------------")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print(
        "\nDEMO ONLY: These results use "
        "synthetically generated data."
    )

    print(
        "No models or experimental results "
        "were saved as real research artifacts."
    )