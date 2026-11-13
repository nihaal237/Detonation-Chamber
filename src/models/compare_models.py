import argparse

import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from train_random_forest import (
    create_demo_dataset,
    split_dataset,
    RANDOM_SEED
)

from compare_feature_groups import (
    FEATURE_GROUPS,
    evaluate_classifier
)


# --------------------------------------------------
# 1. Define ML Algorithms
# --------------------------------------------------

def create_models():

    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_SEED
        )
    }

    return models


# --------------------------------------------------
# 2. Train and Evaluate Models
# --------------------------------------------------

def compare_models(train_df, test_df):

    results = []

    for algorithm_name, model_template in create_models().items():

        for feature_group, feature_columns in FEATURE_GROUPS.items():

            print(
                f"\nTraining {algorithm_name} "
                f"using {feature_group} features..."
            )

            X_train = train_df[feature_columns]

            y_train = train_df["label"]

            # Create a new model for each experiment

            from sklearn.base import clone

            model = clone(model_template)

            model.fit(
                X_train,
                y_train
            )

            metrics = evaluate_classifier(
                model,
                test_df,
                feature_columns
            )

            results.append({

                "algorithm": algorithm_name,

                "feature_group": feature_group,

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

    return pd.DataFrame(results)


# --------------------------------------------------
# 3. Display Comparison
# --------------------------------------------------

def display_results(results):

    print("\n--------------------------------")
    print("MODEL COMPARISON RESULTS")
    print("--------------------------------")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nMacro-F1 Comparison:")

    comparison = results.pivot(
        index="algorithm",
        columns="feature_group",
        values="macro_f1"
    )

    print(
        comparison.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


# --------------------------------------------------
# 4. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare malware classification models"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic demonstration data"
    )

    args = parser.parse_args()

    if not args.demo:

        parser.error(
            "Only --demo is supported currently."
        )

    print("\nCreating synthetic demonstration dataset...")

    df = create_demo_dataset()

    print("Total records:", len(df))

    print(
        "Unique samples:",
        df["sample_id"].nunique()
    )

    # Use only the 180-second observation window

    df = df[
        df["observation_window"] == 180
    ].copy()

    print("\nObservation window: 180 seconds")

    print("Selected records:", len(df))

    # Create one shared train/test split

    train_df, test_df = split_dataset(df)

    print(
        "\nTraining samples:",
        train_df["sample_id"].nunique()
    )

    print(
        "Testing samples:",
        test_df["sample_id"].nunique()
    )

    # Train and evaluate all six configurations

    results = compare_models(
        train_df,
        test_df
    )

    display_results(results)

    print(
        "\nDEMO ONLY: Results use artificially "
        "generated data."
    )

    print(
        "No models or results were saved "
        "as real research artifacts."
    )