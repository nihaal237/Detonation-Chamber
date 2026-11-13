import json
import argparse

from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

STATIC_PATH = Path(
    "data/processed/static_dataset.csv"
)

DYNAMIC_PATH = Path(
    "data/processed/dynamic_features.json"
)

OUTPUT_PATH = Path(
    "data/processed/combined_features.csv"
)


# --------------------------------------------------
# 2. Load Static Features
# --------------------------------------------------

def load_static_features(file_path):

    df = pd.read_csv(file_path)

    if "sha256" not in df.columns:
        raise ValueError(
            "Static dataset is missing SHA-256."
        )

    df["sha256"] = (
        df["sha256"]
        .astype(str)
        .str.lower()
    )

    if df["sha256"].duplicated().any():
        raise ValueError(
            "Duplicate SHA-256 values in static dataset."
        )

    return df


# --------------------------------------------------
# 3. Load Dynamic Features
# --------------------------------------------------

def load_dynamic_features(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Dynamic features must be a JSON list."
        )

    df = pd.DataFrame(records)

    required_columns = [
        "sha256",
        "observation_window"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dynamic columns: {missing}"
        )

    df["sha256"] = (
        df["sha256"]
        .astype(str)
        .str.lower()
    )

    duplicate_windows = df.duplicated(
        subset=["sha256", "observation_window"]
    )

    if duplicate_windows.any():
        raise ValueError(
            "Duplicate sample/window combinations found."
        )

    return df


# --------------------------------------------------
# 4. Combine Features Using SHA-256
# --------------------------------------------------

def combine_features(static_df, dynamic_df):

    combined = pd.merge(
        static_df,
        dynamic_df,
        on="sha256",
        how="inner",
        validate="one_to_many",
        suffixes=("_static", "_dynamic")
    )

    return combined


# --------------------------------------------------
# 5. Report Unmatched Samples
# --------------------------------------------------

def report_unmatched(static_df, dynamic_df):

    static_hashes = set(
        static_df["sha256"]
    )

    dynamic_hashes = set(
        dynamic_df["sha256"]
    )

    static_only = static_hashes - dynamic_hashes

    dynamic_only = dynamic_hashes - static_hashes

    print(
        "\nStatic samples without dynamic features:",
        len(static_only)
    )

    print(
        "Dynamic samples without static features:",
        len(dynamic_only)
    )

    return static_only, dynamic_only


# --------------------------------------------------
# 6. Create Harmless Demo Data
# --------------------------------------------------

def create_demo_data():

    demo_hash = "a" * 64

    static_df = pd.DataFrame([
        {
            "sha256": demo_hash,
            "file_size": 100000,
            "number_of_sections": 5,
            "number_of_imports": 20
        }
    ])

    dynamic_df = pd.DataFrame([
        {
            "sha256": demo_hash,
            "observation_window": 10,
            "process_create_count": 1,
            "file_create_count": 1
        },
        {
            "sha256": demo_hash,
            "observation_window": 30,
            "process_create_count": 2,
            "file_create_count": 3
        },
        {
            "sha256": demo_hash,
            "observation_window": 60,
            "process_create_count": 4,
            "file_create_count": 5
        },
        {
            "sha256": demo_hash,
            "observation_window": 180,
            "process_create_count": 6,
            "file_create_count": 10
        }
    ])

    return static_df, dynamic_df


# --------------------------------------------------
# 7. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Combine static and dynamic features"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic demonstration data"
    )

    args = parser.parse_args()

    try:

        if args.demo:

            print("\nUsing synthetic demonstration data.")

            static_df, dynamic_df = create_demo_data()

        else:

            static_df = load_static_features(
                STATIC_PATH
            )

            dynamic_df = load_dynamic_features(
                DYNAMIC_PATH
            )

        print("\nStatic samples:", len(static_df))

        print("Dynamic records:", len(dynamic_df))

        combined = combine_features(
            static_df,
            dynamic_df
        )

        report_unmatched(
            static_df,
            dynamic_df
        )

        print(
            "\nMatched combined records:",
            len(combined)
        )

        if args.demo:

            print("\nDemonstration combined dataset:")

            print(combined.to_string(index=False))

            print(
                "\nDemo data was not saved "
                "to the research dataset."
            )

        elif combined.empty:

            print(
                "\nNo matching SHA-256 hashes found."
            )

            print(
                "No combined dataset was written."
            )

        else:

            OUTPUT_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            combined.to_csv(
                OUTPUT_PATH,
                index=False
            )

            print(
                "\nCombined features saved to:",
                OUTPUT_PATH
            )

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError
    ) as error:

        print("\nIntegration failed:", error)