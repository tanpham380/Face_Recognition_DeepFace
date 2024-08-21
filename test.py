import os
import glob

from core.utils.static_variable import IMAGES_DIR

def delete_images_for_uid( uid: str, base_uid: str):
    # Adjust the pattern to correctly match files associated with the base_uid
    image_pattern = os.path.join(IMAGES_DIR,base_uid ,f"*_{uid}*.png")
    print(f"Deleting images for UID {base_uid} with pattern {image_pattern}")

    # Ensure the pattern is correct and matches files
    matching_files = glob.glob(image_pattern)
    print(f"Matching files: {matching_files}")

    if not matching_files:
        print(f"No files found for pattern: {image_pattern}")

    for file_path in matching_files:
        try:
            print(f"Deleting file {file_path}")
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete file {file_path}: {e}")

# Example usage
delete_images_for_uid("thanhtan2136-1", "thanhtan2136")
