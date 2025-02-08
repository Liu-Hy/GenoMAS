import hashlib
import os
import json
from pathlib import Path
import argparse
from tqdm import tqdm

class DatasetValidator:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        
    def calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def generate_manifest(self, output_path="manifest.json"):
        """Generate manifest file containing path and hash information"""
        manifest = {
            "dataset_name": self.dataset_path.name,
            "files": {}
        }
        
        # First collect all files to show accurate progress
        all_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                file_path = Path(root) / file
                all_files.append(file_path)
        
        # Process files with progress bar
        for file_path in tqdm(all_files, desc="Generating manifest", unit="file"):
            relative_path = str(file_path.relative_to(self.dataset_path))
            relative_path = relative_path.replace("\\", "/")
            manifest["files"][relative_path] = self.calculate_file_hash(file_path)
        
        # Write manifest to file
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest
    
    def validate_dataset(self, manifest_path="manifest.json"):
        """Validate dataset against manifest"""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        errors = []
        missing_files = []
        
        # Check each file in manifest with progress bar
        manifest_files = list(manifest["files"].items())
        for relative_path, expected_hash in tqdm(manifest_files, desc="Validating files", unit="file"):
            file_path = self.dataset_path / Path(relative_path)
            
            if not file_path.exists():
                missing_files.append(relative_path)
                continue
                
            actual_hash = self.calculate_file_hash(file_path)
            if actual_hash != expected_hash:
                errors.append(f"Hash mismatch for {relative_path}")
        
        # Check for extra files
        print("Checking for extra files...")
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                file_path = Path(root) / file
                relative_path = str(file_path.relative_to(self.dataset_path)).replace("\\", "/")
                if relative_path not in manifest["files"]:
                    errors.append(f"Extra file found: {relative_path}")
        
        return {
            "is_valid": len(errors) == 0 and len(missing_files) == 0,
            "errors": errors,
            "missing_files": missing_files
        }

def main():
    parser = argparse.ArgumentParser(description='Dataset validation tool')
    parser.add_argument('--data-dir', required=True, help='Path to data directory containing input datasets')
    parser.add_argument('--generate', action='store_true', help='Generate manifest files')
    parser.add_argument('--validate', action='store_true', help='Validate datasets against manifest files')
    
    args = parser.parse_args()
    data_path = Path(args.data_dir)
    datasets = ['TCGA', 'GEO']
    
    for dataset in datasets:
        dataset_path = data_path / dataset
        if not dataset_path.exists():
            print(f"Warning: {dataset} directory not found at {dataset_path}")
            continue
            
        manifest_path = f"{dataset.lower()}_manifest.json"
        validator = DatasetValidator(dataset_path)
        
        if args.generate:
            print(f"\nGenerating manifest file for {dataset}: {manifest_path}")
            validator.generate_manifest(manifest_path)
            print(f"{dataset} manifest generation complete")
            
        if args.validate:
            print(f"\nValidating {dataset} dataset...")
            result = validator.validate_dataset(manifest_path)
            
            if result["is_valid"]:
                print(f"{dataset} dataset is valid!")
            else:
                print(f"{dataset} dataset validation failed!")
                if result["missing_files"]:
                    print("\nMissing files:")
                    for file in result["missing_files"]:
                        print(f"  - {file}")
                if result["errors"]:
                    print("\nErrors:")
                    for error in result["errors"]:
                        print(f"  - {error}")

if __name__ == "__main__":
    main()
# Example Usage: python validator.py --data-dir /path/to/data --generate
#                python validator.py --data-dir /path/to/data --validate