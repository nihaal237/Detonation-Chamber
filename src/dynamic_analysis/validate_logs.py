import json
import re
import argparse

from pathlib import Path


# --------------------------------------------------
# 1. Supported Event Types
# --------------------------------------------------

EVENT_TYPES = {
    "process_create",
    "file_create",
    "file_modify",
    "file_delete",
    "registry_set",
    "registry_delete",
    "network_attempt"
}


# --------------------------------------------------
# 2. Validate a Behavioral Record
# --------------------------------------------------

def validate_record(record):

    errors = []

    required_fields = [
        "schema_version",
        "sample_id",
        "sha256",
        "run_status",
        "observation_seconds",
        "monitoring_coverage",
        "events"
    ]

    # Check required fields

    for field in required_fields:

        if field not in record:

            errors.append(
                f"Missing required field: {field}"
            )

    if errors:
        return errors

    # Check schema version

    if record["schema_version"] != "1.0":

        errors.append(
            "Unsupported schema version."
        )

    # Check sample ID

    if not isinstance(record["sample_id"], str) or not record["sample_id"]:

        errors.append(
            "sample_id must be a nonempty string."
        )

    # Validate SHA-256

    sha256 = record["sha256"]

    if (
        not isinstance(sha256, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", sha256)
    ):

        errors.append(
            "Invalid SHA-256 hash."
        )

    # Validate run status

    valid_statuses = {
        "completed",
        "failed",
        "timeout"
    }

    if record["run_status"] not in valid_statuses:

        errors.append(
            "Invalid run_status."
        )

    # Validate observation duration

    duration = record["observation_seconds"]

    valid_duration = (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and 0 < duration <= 3600
    )

    if not valid_duration:

        errors.append(
            "observation_seconds must be "
            "a number between 0 and 3600."
        )

    # Validate monitoring coverage

    coverage = record["monitoring_coverage"]

    if not isinstance(coverage, dict):

        errors.append(
            "monitoring_coverage must be an object."
        )

    else:

        for event_type in EVENT_TYPES:

            if event_type not in coverage:

                errors.append(
                    f"Missing monitoring coverage: {event_type}"
                )

            elif coverage[event_type] not in [
                True,
                False,
                None
            ]:

                errors.append(
                    f"Invalid coverage value: {event_type}"
                )

    # Validate events

    events = record["events"]

    if not isinstance(events, list):

        errors.append(
            "events must be a list."
        )

        return errors

    for index, event in enumerate(events):

        if not isinstance(event, dict):

            errors.append(
                f"Event {index}: must be an object."
            )

            continue

        # Validate event type

        if event.get("type") not in EVENT_TYPES:

            errors.append(
                f"Event {index}: invalid event type."
            )

        # Validate timestamp

        timestamp = event.get("t_seconds")

        valid_timestamp = (
            isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and timestamp >= 0
        )

        if valid_timestamp and valid_duration:

            valid_timestamp = timestamp <= duration

        if not valid_timestamp:

            errors.append(
                f"Event {index}: invalid timestamp."
            )

        # Validate source

        if not isinstance(event.get("source"), str) or not event.get("source"):

            errors.append(
                f"Event {index}: missing source."
            )

        # Validate details

        if not isinstance(event.get("details"), dict):

            errors.append(
                f"Event {index}: details must be an object."
            )

    return errors


# --------------------------------------------------
# 3. Load JSON
# --------------------------------------------------

def load_record(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# --------------------------------------------------
# 4. Harmless Demonstration Record
# --------------------------------------------------

def create_demo_record():

    return {

        "schema_version": "1.0",

        "sample_id": "demo_001",

        "sha256": "a" * 64,

        "run_status": "completed",

        "observation_seconds": 180,

        "monitoring_coverage": {
            "process_create": True,
            "file_create": True,
            "file_modify": True,
            "file_delete": True,
            "registry_set": True,
            "registry_delete": True,
            "network_attempt": None
        },

        "events": [

            {
                "t_seconds": 1.2,
                "type": "process_create",
                "source": "sysmon",
                "details": {
                    "image": "demo.exe"
                }
            },

            {
                "t_seconds": 3.5,
                "type": "file_create",
                "source": "sysmon",
                "details": {
                    "path": "<redacted>"
                }
            },

            {
                "t_seconds": 7.1,
                "type": "registry_set",
                "source": "procmon",
                "details": {
                    "key": "<redacted>"
                }
            },
            {
    "t_seconds": 45.0,
    "type": "file_delete",
    "source": "procmon",
    "details": {
        "path": "<redacted>"
    }
            }
        ]
    }


# --------------------------------------------------
# 5. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Validate behavioral analysis logs"
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a behavioral JSON record"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Validate a harmless demonstration record"
    )

    args = parser.parse_args()

    if args.demo:

        record = create_demo_record()

        print("\nUsing demonstration data.")

    elif args.input:

        try:

            record = load_record(args.input)

        except (OSError, json.JSONDecodeError) as error:

            parser.error(str(error))

    else:

        parser.error(
            "Provide --input or --demo."
        )

    if not isinstance(record, dict):

        parser.error(
            "The JSON root must be an object."
        )

    errors = validate_record(record)

    print("\n--------------------------------")
    print("BEHAVIORAL LOG VALIDATION")
    print("--------------------------------")

    if errors:

        print("\nValidation failed.")

        for error in errors:

            print("-", error)

        raise SystemExit(1)

    else:

        print("\nValidation passed!")

        print("Sample ID:", record["sample_id"])

        print(
            "Observation duration:",
            record["observation_seconds"],
            "seconds"
        )

        print(
            "Recorded events:",
            len(record["events"])
        )