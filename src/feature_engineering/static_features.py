import json
import os
import statistics


# --------------------------------------------------
# 1. Define API Categories
# --------------------------------------------------

REGISTRY_APIS = { #examples of Windows API functions associated with Registry operations.
    "RegOpenKeyExA",
    "RegOpenKeyExW",
    "RegCreateKeyExA",
    "RegCreateKeyExW",
    "RegCreateKeyA",
    "RegCreateKeyW",
    "RegSetValueExA",
    "RegSetValueExW",
    "RegDeleteKeyA",
    "RegDeleteKeyW",
}

NETWORK_APIS = {
    "connect",
    "send",
    "recv",
    "WSAConnect",
    "InternetOpenA",
    "InternetOpenW",
    "InternetConnectA",
    "InternetConnectW",
    "HttpSendRequestA",
    "HttpSendRequestW",
    "WinHttpConnect",
    "WinHttpSendRequest",
}

PROCESS_APIS = {
    "CreateProcessA",
    "CreateProcessW",
    "OpenProcess",
    "TerminateProcess",
    "CreateRemoteThread",
    "WriteProcessMemory",
    "ReadProcessMemory",
    "VirtualAllocEx",
}

FILE_APIS = {
    "CreateFileA",
    "CreateFileW",
    "WriteFile",
    "ReadFile",
    "DeleteFileA",
    "DeleteFileW",
    "MoveFileA",
    "MoveFileW",
    "CopyFileA",
    "CopyFileW",
}


# --------------------------------------------------
# 2. Calculate Section Statistics
# --------------------------------------------------

def calculate_section_features(sections):

    entropies = [
        section["entropy"]
        for section in sections
    ]

    if not entropies:

        return {
            "average_section_entropy": 0.0,
            "maximum_section_entropy": 0.0,
            "minimum_section_entropy": 0.0,
            "high_entropy_sections": 0
        }

    return {

        "average_section_entropy": round( #calculates the average entropy across the executable's PE sections.
            statistics.mean(entropies), 4
        ),

        "maximum_section_entropy": round(
            max(entropies), 4
        ),

        "minimum_section_entropy": round(
            min(entropies), 4
        ),

        "high_entropy_sections": sum(
            entropy >= 7.0
            for entropy in entropies
        )
    }


# --------------------------------------------------
# 3. Calculate API Import Features
# --------------------------------------------------

def calculate_api_features(imported_functions):

    imported_set = {
        function.lower()
        for function in imported_functions
    }

    def count_matches(api_category):

        return sum(
            api.lower() in imported_set
            for api in api_category
        )

    return {

        "registry_api_count": count_matches(REGISTRY_APIS),

        "network_api_count": count_matches(NETWORK_APIS),

        "process_api_count": count_matches(PROCESS_APIS),

        "file_api_count": count_matches(FILE_APIS),

        "imports_registry_api": int(
            count_matches(REGISTRY_APIS) > 0
        ),

        "imports_network_api": int(
            count_matches(NETWORK_APIS) > 0
        ),

        "imports_process_api": int(
            count_matches(PROCESS_APIS) > 0
        ),

        "imports_file_api": int(
            count_matches(FILE_APIS) > 0
        )
    }


# --------------------------------------------------
# 4. Combine All Static Features
# --------------------------------------------------

def engineer_static_features(raw_features):

    features = {}

    # Identification

    features["sha256"] = raw_features["sha256"]

    # Basic numerical features

    features["file_size"] = raw_features["file_size"]

    features["machine_type"] = raw_features["machine_type"]

    features["number_of_sections"] = raw_features[
        "number_of_sections"
    ]

    features["number_of_imports"] = raw_features[
        "number_of_imports"
    ]

    features["entry_point"] = raw_features["entry_point"]

    # Section-based features

    section_features = calculate_section_features(
        raw_features["sections"]
    )

    features.update(section_features) #method adds the section statistics to our main features dictionary.

    # API-based features

    api_features = calculate_api_features(
        raw_features["imported_functions"]
    )

    features.update(api_features)

    return features


# --------------------------------------------------
# 5. Load JSON
# --------------------------------------------------

def load_json(file_path):

    with open(file_path, "r", encoding="utf-8") as file:

        return json.load(file)


# --------------------------------------------------
# 6. Save Engineered Features
# --------------------------------------------------

def save_json(data, output_path):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(data, file, indent=4)


# --------------------------------------------------
# 7. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    input_path = "data/processed/static_features.json"

    output_path = "data/processed/static_ml_features.json"

    raw_features = load_json(input_path)

    engineered_features = engineer_static_features(
        raw_features
    )

    save_json(engineered_features, output_path)

    print("\nStatic feature engineering completed!")

    print("Output saved to:", output_path)

    print("\nEngineered Features:")

    print(json.dumps(engineered_features, indent=4))