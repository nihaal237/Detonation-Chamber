import argparse

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    f1_score
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_SEED = 42

OBSERVATION_WINDOWS = [10, 30, 60, 180]

DYNAMIC_FEATURES = [
    "process_create_count",
    "file_create_count",
    "registry_set_count"
]


# --------------------------------------------------
# 2. Generate Cumulative Synthetic Data
# --------------------------------------------------

def create_demo_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    records = []

    for sample_index in range(100):

        sample_id = f"synthetic_{sample_index:04d}"

        label = sample_index % 2

        # Start all event counts at zero

        process_total = 0

        file_total = 0

        registry_total = 0

        previous_window = 0

        for window in OBSERVATION_WINDOWS:

            interval_duration = window - previous_window

            previous_window = window

            # Artificial event rates.
            # Malware-labeled samples have higher
            # expected activity in this demonstration.

            process_rate = 0.03 + label * 0.05

            file_rate = 0.05 + label * 0.12

            registry_rate = 0.01 + label * 0.04

            # Generate new events for this interval

            new_processes = rng.poisson(
                process_rate * interval_duration
            )

            new_files = rng.poisson(
                file_rate * interval_duration
            )

            new_registry_events = rng.poisson(
                registry_rate * interval_duration
            )

            # Add to cumulative counts

            process_total += new_processes

            file_total += new_files

            registry_total += new_registry_events

            records.append({

                "sample_id": sample_id,

                "label": label,

                "observation_window": window,

                "process_create_count": int(process_total),

                "file_create_count": int(file_total),

                "registry_set_count": int(registry_total)
            })

    return pd.DataFrame(records)


# --------------------------------------------------
# 3. Create Shared Sample-Level Split
# --------------------------------------------------

def split_sample_ids(df):

    sample_labels = (
        df.groupby("sample_id")["label"]
        .first()
        .reset_index()
    )

    train_ids, test_ids = train_test_split(
        sample_labels["sample_id"],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=sample_labels["label"]
    )

    return set(train_ids), set(test_ids)


# --------------------------------------------------
# 4. Define Models
# --------------------------------------------------

def create_models():

    return {

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


# --------------------------------------------------
# 5. Run Observation-Window Experiment
# --------------------------------------------------

def run_experiment(df):

    train_ids, test_ids = split_sample_ids(df)

    print(
        "\nTraining samples:",
        len(train_ids)
    )

    print(
        "Testing samples:",
        len(test_ids)
    )

    assert train_ids.isdisjoint(test_ids)

    results = []

    for window in OBSERVATION_WINDOWS:

        print(
            f"\nEvaluating {window}-second window..."
        )

        window_df = df[
            df["observation_window"] == window
        ].copy()

        train_df = window_df[
            window_df["sample_id"].isin(train_ids)
        ]

        test_df = window_df[
            window_df["sample_id"].isin(test_ids)
        ]

        X_train = train_df[DYNAMIC_FEATURES]

        y_train = train_df["label"]

        X_test = test_df[DYNAMIC_FEATURES]

        y_test = test_df["label"]

        for model_name, model in create_models().items():

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            macro_f1 = f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )

            results.append({

                "window_seconds": window,

                "model": model_name,

                "accuracy": accuracy,

                "macro_f1": macro_f1
            })

            print(
                f"  {model_name}: "
                f"Accuracy={accuracy:.4f}, "
                f"Macro-F1={macro_f1:.4f}"
            )

    return pd.DataFrame(results)


# --------------------------------------------------
# 6. Verify Cumulative Counts
# --------------------------------------------------

def verify_cumulative_counts(df):

    sorted_df = df.sort_values(
        ["sample_id", "observation_window"]
    )

    for feature in DYNAMIC_FEATURES:

        differences = (
            sorted_df.groupby("sample_id")[feature]
            .diff()
        )

        if (differences.dropna() < 0).any():

            raise ValueError(
                f"Non-cumulative values found in {feature}"
            )

    print("\nCumulative count validation: PASSED")


# --------------------------------------------------
# 7. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Observation-window evaluation"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use cumulative synthetic data"
    )

    args = parser.parse_args()

    if not args.demo:

        parser.error(
            "Only --demo is currently supported."
        )

    print("\nCreating cumulative synthetic dataset...")

    df = create_demo_dataset()

    print("Total records:", len(df))

    print(
        "Unique samples:",
        df["sample_id"].nunique()
    )

    verify_cumulative_counts(df)

    results = run_experiment(df)

    print("\n--------------------------------")
    print("OBSERVATION WINDOW RESULTS")
    print("--------------------------------")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nMacro-F1 Comparison:")

    comparison = results.pivot(
        index="window_seconds",
        columns="model",
        values="macro_f1"
    )

    print(
        comparison.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print(
        "\nDEMO ONLY: These results use "
        "artificially generated behavior."
    )

    print(
        "No models or research results were saved."
    )