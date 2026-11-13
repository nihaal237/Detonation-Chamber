import argparse

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

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

MALWARE_FAMILIES = [
    "RemcosRAT",
    "AgentTesla",
    "Smoke Loader",
    "LockBit",
    "CoinMiner"
]

FEATURE_COLUMNS = [
    "process_create_count",
    "file_create_count",
    "registry_set_count",
    "file_delete_count"
]


# --------------------------------------------------
# 2. Generate Synthetic Demonstration Data
# --------------------------------------------------

def create_demo_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    records = []

    samples_per_family = 50

    benign_samples = 100

    # ----------------------------------------------
    # Generate Benign Samples
    # ----------------------------------------------

    for index in range(benign_samples):

        records.append({

            "sample_id": f"benign_{index:04d}",

            "family": "Benign",

            "label": 0,

            "process_create_count": int(
                rng.poisson(2)
            ),

            "file_create_count": int(
                rng.poisson(3)
            ),

            "registry_set_count": int(
                rng.poisson(1)
            ),

            "file_delete_count": int(
                rng.poisson(1)
            )
        })

    # ----------------------------------------------
    # Generate Malware Families
    # ----------------------------------------------

    family_rates = {

        "RemcosRAT": [6, 5, 7, 2],

        "AgentTesla": [4, 7, 4, 2],

        "Smoke Loader": [8, 9, 3, 4],

        "LockBit": [5, 16, 5, 12],

        "CoinMiner": [7, 4, 6, 2]
    }

    for family in MALWARE_FAMILIES:

        rates = family_rates[family]

        for index in range(samples_per_family):

            records.append({

                "sample_id": (
                    f"{family}_{index:04d}"
                ),

                "family": family,

                "label": 1,

                "process_create_count": int(
                    rng.poisson(rates[0])
                ),

                "file_create_count": int(
                    rng.poisson(rates[1])
                ),

                "registry_set_count": int(
                    rng.poisson(rates[2])
                ),

                "file_delete_count": int(
                    rng.poisson(rates[3])
                )
            })

    return pd.DataFrame(records)


# --------------------------------------------------
# 3. Split Benign Samples
# --------------------------------------------------

def split_benign_samples(df):

    benign_df = df[
        df["label"] == 0
    ].copy()

    train_benign, test_benign = train_test_split(
        benign_df,
        test_size=0.2,
        random_state=RANDOM_SEED
    )

    return train_benign, test_benign


# --------------------------------------------------
# 4. Train Binary Classifier
# --------------------------------------------------

def train_classifier(train_df):

    X_train = train_df[FEATURE_COLUMNS]

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
# 5. Evaluate One Unseen Family
# --------------------------------------------------

def evaluate_family(
    model,
    unseen_malware,
    test_benign
):

    test_df = pd.concat(
        [unseen_malware, test_benign],
        ignore_index=True
    )

    X_test = test_df[FEATURE_COLUMNS]

    y_test = test_df["label"]

    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    malware_detection_rate = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

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

        "unseen_detection_rate": malware_detection_rate,

        "false_positive_rate": false_positive_rate
    }


# --------------------------------------------------
# 6. Run Leave-One-Family-Out Experiment
# --------------------------------------------------

def run_lofo(df):

    train_benign, test_benign = split_benign_samples(
        df
    )

    results = []

    for held_out_family in MALWARE_FAMILIES:

        print(
            f"\nHolding out family: {held_out_family}"
        )

        # Malware from all other families

        train_malware = df[
            (df["label"] == 1)
            & (df["family"] != held_out_family)
        ].copy()

        # Completely unseen malware family

        unseen_malware = df[
            (df["label"] == 1)
            & (df["family"] == held_out_family)
        ].copy()

        # Construct training dataset

        train_df = pd.concat(
            [train_malware, train_benign],
            ignore_index=True
        )

        # Verify no sample identity overlap

        train_ids = set(train_df["sample_id"])

        test_ids = set(
            pd.concat(
                [unseen_malware, test_benign]
            )["sample_id"]
        )

        assert train_ids.isdisjoint(test_ids)

        # Train binary model

        model = train_classifier(train_df)

        # Evaluate unseen family

        metrics = evaluate_family(
            model,
            unseen_malware,
            test_benign
        )

        results.append({

            "held_out_family": held_out_family,

            "unseen_detection_rate":
                metrics["unseen_detection_rate"],

            "false_positive_rate":
                metrics["false_positive_rate"],

            "accuracy": metrics["accuracy"],

            "macro_f1": metrics["macro_f1"]
        })

        print(
            "Unseen malware detection:",
            f"{metrics['unseen_detection_rate']:.4f}"
        )

        print(
            "False-positive rate:",
            f"{metrics['false_positive_rate']:.4f}"
        )

    return pd.DataFrame(results)


# --------------------------------------------------
# 7. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Leave-one-family-out evaluation"
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

    print("\nCreating synthetic LOFO dataset...")

    df = create_demo_dataset()

    print("Total samples:", len(df))

    print("\nFamily distribution:")

    print(
        df["family"].value_counts()
    )

    results = run_lofo(df)

    print("\n--------------------------------")
    print("LEAVE-ONE-FAMILY-OUT RESULTS")
    print("--------------------------------")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print(
        "\nDEMO ONLY: These results use "
        "artificially generated samples."
    )

    print(
        "No models or research results were saved."
    )