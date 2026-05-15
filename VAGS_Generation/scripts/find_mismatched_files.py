# python scripts/find_mismatched_files.py /path/to/datasets/coco14/val
import os
import argparse


def find_mismatched_files(folder_path):
    """
    Find files in the given folder that have only PNG or only TXT versions,
    but not both.
    
    Args:
        folder_path (str): Path to the folder containing files
        
    Returns:
        tuple: (png_only, txt_only) lists of IDs with only one file type
    """
    # Get all PNG and TXT files
    png_files = set()
    txt_files = set()
    
    # Walk through all files in the folder
    for filename in os.listdir(folder_path):
        # Skip directories and hidden files
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path) or filename.startswith('.'):
            continue
            
        # Split filename and extension
        file_parts = os.path.splitext(filename)
        if len(file_parts) != 2:
            continue
            
        file_id, extension = file_parts
        
        # Add to appropriate set based on extension
        if extension.lower() == '.jpg':
            png_files.add(file_id)
        elif extension.lower() == '.txt':
            txt_files.add(file_id)
    
    # Find mismatches by comparing sets
    png_only = list(png_files - txt_files)  # IDs that have PNG but not TXT
    txt_only = list(txt_files - png_files)  # IDs that have TXT but not PNG
    
    return png_only, txt_only


def main():
    parser = argparse.ArgumentParser(description='Find files with missing PNG or TXT counterparts')
    parser.add_argument('folder', help='Input folder containing PNG and TXT files')
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        return 1
        
    png_only, txt_only = find_mismatched_files(args.folder)
    
    if not png_only and not txt_only:
        print("All files have matching counterparts!")
        return 0
        
    print(f"Found {len(png_only)} files with PNG only (missing TXT):")
    for file_id in sorted(png_only):
        print(f"  - {file_id}.jpg")
    
    print(f"\nFound {len(txt_only)} files with TXT only (missing PNG):")
    for file_id in sorted(txt_only):
        print(f"  - {file_id}.txt")

    print(len(png_only), len(txt_only))
        
    return 0


if __name__ == "__main__":
    exit(main())