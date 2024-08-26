import os
import datetime
from typing import Any, Tuple

import os
import datetime
from typing import Any

from core.utils.logging import get_logger
from core.utils.static_variable import IMAGES_DIR
import glob
logger = get_logger()



def delete_directory_if_empty(save_dir: str) -> bool:
    remaining_images = [f for f in os.listdir(save_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not remaining_images:
        os.rmdir(save_dir)
        logger.info(f"Deleted empty directory: {save_dir}")
        return True
    return False

def extract_base_identity(identity):
    # Split the identity by '_' and take the last part without the augmentation part
    identity = os.path.splitext(os.path.basename(identity))[0]
    identity = "_".join(identity.split('_')[2:]).split('_')[0]

    return identity


def delete_images_for_uid(uid: str, base_uid: str):
    # Adjust the pattern to correctly match files associated with the base_uid and uid
    image_pattern = os.path.join(IMAGES_DIR, base_uid, f"*_{uid}*.png")

    # Ensure the pattern is correct and matches files
    matching_files = glob.glob(image_pattern)

    if not matching_files:
        logger.info(f"No files found for pattern")

    for file_path in matching_files:
        try:
            os.remove(file_path)
            logger.info(f"Deleted file {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")


def save_image(image: Any, uid: str, base_dir: str, prefix: str, anti_spoofing: bool = False):
    """
    Save the given image to a directory with a specific format. If the directory contains more than
    56 images (8 original and 48 augmented), the oldest augmented images will be deleted.

    Parameters:
    - image: The image object to be saved.
    - uid: The UID or identifier for the directory.
    - base_dir: The base directory where the UID directory will be created.
    - prefix: The prefix to use for the filename (e.g., "query" or UID).
    - anti_spoofing: Boolean flag indicating if the image is flagged for anti-spoofing.

    Returns:
    - The path where the image was saved.
    - The directory where the image was saved.
    """
    # Keep only the part of UID before the '-'
    base_uid = uid.split('-')[0]

    # Determine the target directory
    target_dir = "anti_spoofing" if anti_spoofing else base_uid

    # Create the target directory if it doesn't exist
    save_dir = os.path.join(base_dir, target_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the filename in the format date_UID.png
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"{date_str}_{prefix}.png"
    image_path = os.path.join(save_dir, image_filename)

    # Save the image
    image.save(image_path)

    # Check the number of original and augmented images
    original_images = sorted([f for f in os.listdir(
        save_dir) if 'aug' not in f], key=lambda x: os.path.getmtime(os.path.join(save_dir, x)))
    augmented_images = sorted([f for f in os.listdir(
        save_dir) if 'aug' in f], key=lambda x: os.path.getmtime(os.path.join(save_dir, x)))

    for old_image in original_images[:-4]:
        os.remove(os.path.join(save_dir, old_image))
    for old_image in augmented_images[:-20]:
        os.remove(os.path.join(save_dir, old_image))

    return image_path, save_dir


def delete_images(directory: str):
    """
    Delete all images in the specified directory.

    Parameters:
    - directory: The directory where images will be deleted.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
