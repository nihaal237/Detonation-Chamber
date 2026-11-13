import json
import argparse

from pathlib import Path

from validate_logs import (
    EVENT_TYPES,
    validate_record,
    create_demo_record
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

OBSERVATION_WINDOWS = [10, 30, 60, 180]

OUTPUT_DIR = Path("data/processed")


# --------------------------------------------------
# 2. Extract Features for One Time Window
# --------------------------------------------------

def extract_window_features(record, window_seconds):

    events = record["events"]

    coverage = record["monitoring_coverage"]

    # Only include events observed within the window

    window_events = [
        event
        for event in events
        if event["t_seconds"] <= window_seconds
    ]

    features = {

        "sample_id": record["sample_id"],

        "sha256": record["sha256"],

        "observation_window": window_seconds,

        "run_status": record["run_status"]
    }

    # Count events by type

    for event_type in sorted(EVENT_TYPES):

        feature_name = f"{event_type}_count"

        # Missing monitoring coverage means
        # the event count is unknown.

        if coverage[event_type] is not True:

            features[feature_name] = None

            continue

        count = sum(
            event["type"] == event_type
            for event in window_events
        )

        features[feature_name] = count

    return features


# --------------------------------------------------
# 3. Extract Features for All Windows
# --------------------------------------------------

def extract_all_windows(record):

    errors = validate_record(record)

    if errors:

        raise ValueError(
            "Invalid behavioral record:\n"
            + "\n".join(errors)
        )

    duration = record["observation_seconds"]

    results = []

    for window in OBSERVATION_WINDOWS:

        # Do not invent observations beyond
        # the actual monitoring duration.

        if window > duration:
            continue

        features = extract_window_features(
            record,
            window
        )

        results.append(features)

    return results


# --------------------------------------------------
# 4. Load Behavioral JSON
# --------------------------------------------------

def load_record(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# --------------------------------------------------
# 5. Save Extracted Features
# --------------------------------------------------

def save_features(features, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            features,
            file,
            indent=4
        )


# --------------------------------------------------
# 6. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract dynamic behavioral features"
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Path to behavioral JSON"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use harmless demonstration data"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "dynamic_features.json",
        help="Output JSON path"
    )

    args = parser.parse_args()

    if args.demo:

        print("\nUsing demonstration data.")

        record = create_demo_record()

    elif args.input:

        try:

            record = load_record(args.input)

        except (OSError, json.JSONDecodeError) as error:

            parser.error(str(error))

    else:

        parser.error(
            "Provide --input or --demo."
        )

    try:

        results = extract_all_windows(record)

    except (ValueError, TypeError) as error:

        parser.error(str(error))

    save_features(
        results,
        args.output
    )

    print("\n--------------------------------")
    print("DYNAMIC FEATURE EXTRACTION")
    print("--------------------------------")

    print("\nSample:", record["sample_id"])

    print("Windows processed:", len(results))

    for features in results:

        print(
            f"\nWindow: "
            f"{features['observation_window']} seconds"
        )

        for key, value in features.items():

            if key.endswith("_count"):

                print(f"  {key}: {value}")

    print("\nFeatures saved to:", args.output)