import os
import sys
import csv
import argparse

from pathlib import Path


# --------------------------------------------------
# 1. Import Our Existing Modules
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(SRC_DIR))

from static_analysis.extract_pe_features import extract_pe_features #extracts information from PE files.

from feature_engineering.static_features import engineer_static_features #converts that information into numerical features.


# --------------------------------------------------
# 2. Find Executable Files
# --------------------------------------------------

def find_executables(folder_path):

    executable_files = []

    for file_name in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):

            if file_name.lower().endswith(".exe"):

                executable_files.append(file_path)

    return sorted(executable_files)


# --------------------------------------------------
# 3. Analyze Multiple Executables
# --------------------------------------------------

def batch_extract(folder_path):

    executable_files = find_executables(folder_path)

    dataset = []

    failed_files = []

    print(f"\nFound {len(executable_files)} executable files.\n")

    for index, file_path in enumerate(executable_files, start=1):

        file_name = os.path.basename(file_path)

        print(
            f"[{index}/{len(executable_files)}] "
            f"Analyzing: {file_name}"
        )

        try:

            # Step A: Extract raw PE features

            raw_features = extract_pe_features(file_path)

            # Step B: Convert them into ML features

            ml_features = engineer_static_features(raw_features)

            # Keep filename for identification

            ml_features["file_name"] = file_name

            # Add successful result to dataset

            dataset.append(ml_features)

            print("  Status: Success")

        except Exception as error:

            failed_files.append({
                "file_name": file_name,
                "error": str(error)
            })

            print(f"  Status: Failed - {error}")

    return dataset, failed_files


# --------------------------------------------------
# 4. Save Dataset as CSV
# --------------------------------------------------

def save_dataset(dataset, output_path):

    if not dataset:

        print("No valid PE files were processed.")
        return

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    field_names = list(dataset[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=field_names
        )

        writer.writeheader()

        writer.writerows(dataset)


# --------------------------------------------------
# 5. Save Failed Files
# --------------------------------------------------

def save_failures(failed_files, output_path):

    if not failed_files:
        return

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["file_name", "error"]
        )

        writer.writeheader()

        writer.writerows(failed_files)


# --------------------------------------------------
# 6. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Batch Static PE Feature Extraction"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing executable files"
    )

    args = parser.parse_args()

    input_folder = args.input

    if not os.path.isdir(input_folder):

        print("Error: Input folder does not exist.")

        sys.exit(1)

    dataset, failed_files = batch_extract(input_folder)

    dataset_path = "data/processed/static_dataset.csv"

    failures_path = "reports/static_failures.csv"

    save_dataset(dataset, dataset_path)

    save_failures(failed_files, failures_path)

    print("\n--------------------------------")
    print("Batch Analysis Completed")
    print("--------------------------------")

    print("Successful:", len(dataset))

    print("Failed:", len(failed_files))

    if dataset:
        print("Dataset saved to:", dataset_path)

    if failed_files:
        print("Failure report saved to:", failures_path)