import os
import pandas as pd


# --------------------------------------------------
# 1. Dataset Configuration
# --------------------------------------------------

DATASET_PATH = "data/processed/static_dataset.csv"

REPORT_PATH = "reports/dataset_validation.txt"


# Columns used to identify samples
IDENTIFIER_COLUMNS = [
    "sha256",
    "file_name"
]


# Numerical features define the features our dataset must contain. //Because our future ML model will expect the same columns during training and prediction. If one column is missing, the model might fail or receive incorrect input.
NUMERIC_COLUMNS = [
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


# Features that must be nonnegative
NONNEGATIVE_COLUMNS = [
    "file_size",
    "number_of_sections",
    "number_of_imports",
    "entry_point",
    "high_entropy_sections",
    "registry_api_count",
    "network_api_count",
    "process_api_count",
    "file_api_count"
]


# Binary features
BINARY_COLUMNS = [
    "imports_registry_api",
    "imports_network_api",
    "imports_process_api",
    "imports_file_api"
]


# --------------------------------------------------
# 2. Validate Dataset
# --------------------------------------------------

def validate_dataset(dataset_path):

    print("\nLoading dataset...")

    df = pd.read_csv(dataset_path)

    errors = []
    warnings = []

    print(f"Total samples: {len(df)}")

    # ----------------------------------------------
    # Check 1: Required Columns
    # ----------------------------------------------

    required_columns = IDENTIFIER_COLUMNS + NUMERIC_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        errors.append(
            f"Missing required columns: {missing_columns}"
        )

        return df, errors, warnings

    # ----------------------------------------------
    # Check 2: Missing Values
    # ----------------------------------------------

    missing_values = df[required_columns].isnull().sum()

    for column, count in missing_values.items():

        if count > 0:

            errors.append(
                f"Column '{column}' has {count} missing values."
            )

    # ----------------------------------------------
    # Check 3: Duplicate Samples
    # ----------------------------------------------

    duplicate_count = df.duplicated(
        subset=["sha256"]
    ).sum()

    if duplicate_count > 0:

        warnings.append(
            f"Found {duplicate_count} duplicate SHA-256 records."
        )

    # ----------------------------------------------
    # Check 4: Numerical Features
    # ----------------------------------------------

    for column in NUMERIC_COLUMNS:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_count = (
            converted.isna() & df[column].notna()
        ).sum()

        if invalid_count > 0:

            errors.append(
                f"Column '{column}' contains "
                f"{invalid_count} non-numeric values."
            )

        df[column] = converted

    # ----------------------------------------------
    # Check 5: Entropy Range
    # ----------------------------------------------

    entropy_columns = [
        "average_section_entropy",
        "maximum_section_entropy",
        "minimum_section_entropy"
    ]

    for column in entropy_columns:

        invalid = (
            (df[column] < 0) |
            (df[column] > 8)
        )

        if invalid.any():

            errors.append(
                f"Column '{column}' contains "
                f"{invalid.sum()} invalid entropy values."
            )

    # ----------------------------------------------
    # Check 6: Nonnegative Values
    # ----------------------------------------------

    for column in NONNEGATIVE_COLUMNS:

        invalid = df[column] < 0

        if invalid.any():

            errors.append(
                f"Column '{column}' contains "
                f"{invalid.sum()} negative values."
            )

    # ----------------------------------------------
    # Check 7: Binary Features
    # ----------------------------------------------

    for column in BINARY_COLUMNS:

        invalid = (
            df[column].notna() &
            ~df[column].isin([0, 1])
        )

        if invalid.any():

            errors.append(
                f"Column '{column}' contains "
                f"values other than 0 or 1."
            )

    # ----------------------------------------------
    # Check 8: Dataset Size
    # ----------------------------------------------

    if len(df) < 30:

        warnings.append(
            "Dataset contains fewer than 30 samples. "
            "This is insufficient for meaningful "
            "malware classification experiments."
        )

    return df, errors, warnings


# --------------------------------------------------
# 3. Generate Validation Report
# --------------------------------------------------

def generate_report(df, errors, warnings):

    report = []

    report.append("DETONATION CHAMBER")
    report.append("STATIC DATASET VALIDATION REPORT")
    report.append("=" * 45)

    report.append(f"Total Samples: {len(df)}")
    report.append(f"Total Features: {len(NUMERIC_COLUMNS)}")

    report.append("")

    report.append("ERRORS")
    report.append("-" * 45)

    if errors:

        report.extend(errors)

    else:

        report.append("No errors detected.")

    report.append("")

    report.append("WARNINGS")
    report.append("-" * 45)

    if warnings:

        report.extend(warnings)

    else:

        report.append("No warnings detected.")

    report.append("")

    if errors:

        report.append("VALIDATION STATUS: FAILED")

    else:

        report.append("VALIDATION STATUS: PASSED")

    return "\n".join(report)


# --------------------------------------------------
# 4. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    if not os.path.isfile(DATASET_PATH):

        print("Error: Dataset not found.")

        exit(1)

    df, errors, warnings = validate_dataset(
        DATASET_PATH
    )

    report = generate_report(
        df,
        errors,
        warnings
    )

    print("\n" + report)

    os.makedirs("reports", exist_ok=True)

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print("\nReport saved to:", REPORT_PATH)