import os
import datetime
from typing import Any, Tuple

import os
import datetime
from typing import Any
def extract_base_identity(identity):
    # Split the identity by '_' and take the last part without the augmentation part
    identity = os.path.splitext(os.path.basename(identity))[0]
    identity = "_".join(identity.split('_')[2:]).split('_')[0]

    return identity
def save_image(image: Any, uid: str, base_dir: str, prefix: str, anti_spoofing: bool = False):
    """
    Save the given image to a directory with a specific format. If the directory contains more than
    20 images (4 original and 16 augmented), the oldest augmented images will be deleted.

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
    original_images = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f)) and 'aug' not in f]
    augmented_images = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f)) and 'aug' in f]

    # If the total exceeds 20, delete the oldest augmented images
    if len(original_images) > 4:
        original_images = sorted(
            original_images,
            key=lambda x: os.path.getmtime(os.path.join(save_dir, x))
        )
        # Keep only the 4 most recent original images
        for old_image in original_images[:-4]:
            os.remove(os.path.join(save_dir, old_image))
        original_images = original_images[-4:]

    if len(augmented_images) > 16:
        augmented_images = sorted(
            augmented_images,
            key=lambda x: os.path.getmtime(os.path.join(save_dir, x))
        )
        # Keep only the 16 most recent augmented images
        for old_image in augmented_images[:-16]:
            os.remove(os.path.join(save_dir, old_image))
        augmented_images = augmented_images[-16:]

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
