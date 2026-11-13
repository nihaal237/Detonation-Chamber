import pefile
import hashlib
import math
import json
import os

from collections import Counter


# --------------------------------------------------
# 1. Calculate SHA-256 hash
# --------------------------------------------------

def calculate_sha256(file_path): #generates a cryptographic fingerprint of the file
    sha256 = hashlib.sha256() # If the file's contents change, its SHA-256 hash will almost certainly change.


    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# 2. Calculate Shannon Entropy
# --------------------------------------------------

def calculate_entropy(data): #Entropy measures how unpredictable the bytes are.

    if not data: #Packed, compressed, or encrypted data often has high entropy. But legitimate compressed files can also have high entropy. Our ML model will use entropy alongside other information.
        return 0.0

    byte_counts = Counter(data)

    data_length = len(data)

    entropy = 0.0

    for count in byte_counts.values():

        probability = count / data_length

        entropy -= probability * math.log2(probability)

    return entropy


# --------------------------------------------------
# 3. Extract PE Features
# --------------------------------------------------

def extract_pe_features(file_path): 
    if not os.path.isfile(file_path):
        raise FileNotFoundError("File does not exist.")

    pe = pefile.PE(file_path, fast_load=True) #opens the executable as a PE structure.

    pe.parse_data_directories(
        directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]
        ]
    )

    try:

        features = {}

        # Basic file information

        features["file_name"] = os.path.basename(file_path)

        features["file_size"] = os.path.getsize(file_path)

        features["sha256"] = calculate_sha256(file_path)

        # PE Header Information

        features["machine_type"] = pe.FILE_HEADER.Machine

        features["number_of_sections"] = pe.FILE_HEADER.NumberOfSections

        features["timestamp"] = pe.FILE_HEADER.TimeDateStamp

        features["entry_point"] = pe.OPTIONAL_HEADER.AddressOfEntryPoint

        # Section Information

        sections = []

        for section in pe.sections:

            section_data = section.get_data()

            section_info = {

                "name": section.Name.decode(
                    errors="ignore"
                ).strip("\x00"),

                "virtual_size": section.Misc_VirtualSize,

                "raw_size": section.SizeOfRawData,

                "entropy": round(
                    calculate_entropy(section_data), 4
                )
            }

            sections.append(section_info)

        features["sections"] = sections

        # Imported DLLs and Functions

        imported_dlls = []

        imported_functions = []

        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):

            for entry in pe.DIRECTORY_ENTRY_IMPORT:

                dll_name = entry.dll.decode(
                    errors="ignore"
                )

                imported_dlls.append(dll_name)

                for imported_function in entry.imports:

                    if imported_function.name:

                        function_name = imported_function.name.decode(
                            errors="ignore"
                        )

                        imported_functions.append(function_name)

        features["imported_dlls"] = imported_dlls

        features["imported_functions"] = imported_functions

        features["number_of_imports"] = len(imported_functions)

        return features

    finally:
        pe.close()


# --------------------------------------------------
# 4. Save Features to JSON
# --------------------------------------------------

def save_features(features, output_path):

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(features, file, indent=4) #saves the extracted information in a structured format.


# --------------------------------------------------
# 5. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    file_path = input("Enter the path of the PE file: ").strip().strip('"')

    try:

        features = extract_pe_features(file_path)

        output_path = "data/processed/static_features.json"

        os.makedirs("data/processed", exist_ok=True)

        save_features(features, output_path)

        print("\nStatic analysis completed successfully!")

        print("Features saved to:", output_path)

        print("\nExtracted Features:")

        print(json.dumps(features, indent=4))

    except (FileNotFoundError, pefile.PEFormatError, OSError) as error:

        print("Analysis failed:", error)