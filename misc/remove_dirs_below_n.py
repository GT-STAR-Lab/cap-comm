import os
import argparse
import shutil

# Function to delete nested folders below a specified integer
def delete_nested_folders_below_n_in_specific_dirs(directory, threshold, specific_dirs):
    for root, dirs, _ in os.walk(directory):
        for specific_dir in specific_dirs:
            if specific_dir in dirs:
                dir_path = os.path.join(root, specific_dir)
                for folder_name in os.listdir(dir_path):
                    folder_path = os.path.join(dir_path, folder_name)
                    try:
                        folder_integer = int(folder_name)
                        if folder_integer < threshold:
                            shutil.rmtree(folder_path)
                            print(f"Deleted folder: {folder_path}")
                    except ValueError:
                        pass  # Ignore folders that are not integers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete nested folders below a specified integer threshold under specific directories.")
    parser.add_argument("--threshold", type=int, help="Threshold integer for folder deletion.")
    parser.add_argument("--base-directory", required=True, help="Base directory path.")
    parser.add_argument("--specific-dirs", nargs='+', required=True, help="Specific directory names to search for.")

    args = parser.parse_args()

    if not os.path.isdir(args.base_directory):
        print(f"Error: {args.base_directory} is not a valid directory.")
        exit(1)

    delete_nested_folders_below_n_in_specific_dirs(args.base_directory, args.threshold, args.specific_dirs)

