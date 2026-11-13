import argparse
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

INPUT_PATH = Path(
    "data/processed/malware_candidates.csv"
)

OUTPUT_PATH = Path(
    "data/processed/selected_samples.csv"
)


# --------------------------------------------------
# 2. Load Candidate Dataset
# --------------------------------------------------

def load_candidates():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "sha256",
        "malware_category",
        "query_signature",
        "is_pe_candidate"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Ensure SHA-256 is treated as a string

    df["sha256"] = (
        df["sha256"]
        .astype(str)
        .str.lower()
    )

    return df


# --------------------------------------------------
# 3. Validate Candidates
# --------------------------------------------------

def validate_candidates(df):

    # Keep only PE candidates

    df = df[
        df["is_pe_candidate"] == 1
    ].copy()

    # Remove records with invalid hashes

    valid_hashes = df["sha256"].str.fullmatch(
        r"[0-9a-f]{64}"
    )

    df = df[valid_hashes].copy()

    # Ensure one record per SHA-256

    duplicates = df["sha256"].duplicated()

    if duplicates.any():

        raise ValueError(
            "Duplicate SHA-256 hashes found. "
            "Resolve duplicates before selection."
        )

    # Check for missing categories

    if df["malware_category"].isna().any():

        raise ValueError(
            "Some samples have missing categories."
        )

    return df


# --------------------------------------------------
# 4. Select Candidates
# --------------------------------------------------

def select_samples(df, samples_per_category, seed):

    selected_groups = []

    categories = sorted(
        df["malware_category"].unique()
    )

    for category in categories:

        category_df = df[
            df["malware_category"] == category
        ].copy()

        available = len(category_df)

        print(
            f"\nCategory: {category}"
        )

        print(
            f"Available candidates: {available}"
        )

        if available < samples_per_category:

            raise ValueError(
                f"Not enough candidates for {category}. "
                f"Requested {samples_per_category}, "
                f"available {available}."
            )

        # Sort before sampling for reproducibility

        category_df = category_df.sort_values(
            "sha256"
        )

        # Randomly select candidates

        selected = category_df.sample(
            n=samples_per_category,
            random_state=seed
        )

        selected_groups.append(selected)

        print(
            f"Selected: {len(selected)}"
        )

    return pd.concat(
        selected_groups,
        ignore_index=True
    )


# --------------------------------------------------
# 5. Add Verification Status
# --------------------------------------------------

def add_selection_information(df):

    df = df.copy()

    # Unique identifier for each sample

    df["sample_id"] = df["sha256"]

    # Candidate labels have not been independently verified

    df["verification_status"] = "pending"

    # No sandbox behavior has been collected yet

    df["dynamic_analysis_status"] = "not_started"

    # Track availability of static features

    df["static_analysis_status"] = "not_started"

    return df


# --------------------------------------------------
# 6. Save Selection Manifest
# --------------------------------------------------

def save_selection(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nSelection saved to:",
        OUTPUT_PATH
    )


# --------------------------------------------------
# 7. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Select malware research candidates"
    )

    parser.add_argument(
        "--per-category",
        type=int,
        default=50,
        help="Number of candidates per category"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    if args.per_category < 1:
        parser.error(
            "--per-category must be positive"
        )

    try:

        print("\nLoading malware candidates...")

        df = load_candidates()

        print("Total available records:", len(df))

        df = validate_candidates(df)

        selected = select_samples(
            df,
            args.per_category,
            args.seed
        )

        selected = add_selection_information(
            selected
        )

        save_selection(selected)

        print("\n--------------------------------")
        print("SAMPLE SELECTION SUMMARY")
        print("--------------------------------")

        print("Total selected:", len(selected))

        print("\nSelected candidates per category:")

        print(
            selected["malware_category"].value_counts()
        )

        print("\nVerification status: pending")

        print("No malware binaries were downloaded.")

    except (FileNotFoundError, ValueError) as error:

        print("\nSelection failed:", error)