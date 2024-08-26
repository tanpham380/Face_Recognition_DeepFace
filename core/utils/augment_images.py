from PIL import Image, ImageEnhance, ImageOps
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def shift_image(image, shift):
    width, height = image.size
    shifted_image = Image.new("RGB", (width, height))
    shifted_image.paste(image, (shift[0], shift[1]))
    return shifted_image

def process_image(image, operation, *args):
    if operation == 'rotate':
        return image.rotate(*args, resample=Image.BICUBIC)
    elif operation == 'flip':
        return ImageOps.mirror(image)
    elif operation == 'brightness':
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(*args)
    elif operation == 'shift':
        return shift_image(image, args)

from copy import deepcopy

def augment_image(image_data):
    if hasattr(image_data, 'read'):
        image = Image.open(image_data)
        image.load()
    elif isinstance(image_data, str):
        image = Image.open(image_data)
        image.load()
    else:
        image = image_data

    augmented_images = []
    operations = [
        ('rotate', 5),
        ('rotate', -5),
        ('flip',),
        ('brightness', 0.8),
        # ('shift', 3, 0),
        # ('shift', -3, 0)
    ]

    def process_with_copy(op):
        return process_image(deepcopy(image), *op)

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_with_copy, operations)
        augmented_images.extend(results)

    augmented_images.append(image)

    return augmented_images


