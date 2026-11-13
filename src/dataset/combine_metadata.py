import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

INPUT_DIR = Path("data/samples_metadata")

OUTPUT_DIR = Path("data/processed")

FAMILIES = {
    "remcosrat": "remote_access_trojan",
    "agenttesla": "information_stealer",
    "smoke_loader": "loader",
    "lockbit": "ransomware",
    "coinminer": "cryptocurrency_miner"
}


# --------------------------------------------------
# 2. Load Metadata
# --------------------------------------------------

def load_metadata():

    datasets = []

    for signature, category in FAMILIES.items():

        file_path = INPUT_DIR / f"{signature}_metadata.csv"

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing metadata file: {file_path}"
            )

        df = pd.read_csv(file_path)

        df["query_signature"] = signature

        df["malware_category"] = category

        datasets.append(df)

        print(
            f"{signature}: {len(df)} records loaded"
        )

    return pd.concat(
        datasets,
        ignore_index=True
    )


# --------------------------------------------------
# 3. Filter PE Candidates
# --------------------------------------------------

def filter_pe_candidates(df):

    df = df.copy()

    df = df[
        df["is_pe_candidate"] == 1
    ]

    print(
        "\nPE candidate records:",
        len(df)
    )

    return df


# --------------------------------------------------
# 4. Validate SHA-256
# --------------------------------------------------

def validate_hashes(df):

    df = df.copy()

    valid_hashes = df["sha256"].astype(str).str.fullmatch(
        r"[0-9a-fA-F]{64}"
    )

    invalid_count = (~valid_hashes).sum()

    print("Invalid SHA-256 hashes:", invalid_count)

    return df[valid_hashes].copy()


# --------------------------------------------------
# 5. Identify Cross-Family Conflicts
# --------------------------------------------------

def identify_conflicts(df):

    signature_counts = df.groupby(
        "sha256"
    )["query_signature"].nunique()

    conflicting_hashes = signature_counts[
        signature_counts > 1
    ].index

    conflicts = df[
        df["sha256"].isin(conflicting_hashes)
    ].copy()

    clean = df[
        ~df["sha256"].isin(conflicting_hashes)
    ].copy()

    print(
        "Hashes found under multiple signatures:",
        len(conflicting_hashes)
    )

    return clean, conflicts


# --------------------------------------------------
# 6. Remove Exact Duplicates
# --------------------------------------------------

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates(
        subset=["sha256"]
    ).copy()

    print(
        "Duplicate records removed:",
        before - len(df)
    )

    return df


# --------------------------------------------------
# 7. Save Datasets
# --------------------------------------------------

def save_datasets(clean, conflicts):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    clean_path = OUTPUT_DIR / "malware_candidates.csv"

    conflict_path = OUTPUT_DIR / "metadata_conflicts.csv"

    clean.to_csv(
        clean_path,
        index=False
    )

    conflicts.to_csv(
        conflict_path,
        index=False
    )

    print("\nSaved:", clean_path)

    print("Saved:", conflict_path)


# --------------------------------------------------
# 8. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    print("\nLoading MalwareBazaar metadata...\n")

    df = load_metadata()

    df = filter_pe_candidates(df)

    df = validate_hashes(df)

    clean, conflicts = identify_conflicts(df)

    clean = remove_duplicates(clean)

    save_datasets(clean, conflicts)

    print("\n--------------------------------")
    print("METADATA SUMMARY")
    print("--------------------------------")

    print("\nTotal unique candidates:", len(clean))

    print("\nCandidates per category:")

    print(
        clean["malware_category"].value_counts()
    )

    print("\nMetadata preparation completed!")