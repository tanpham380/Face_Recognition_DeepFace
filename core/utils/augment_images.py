from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from io import BytesIO

def shift_image(image, shift):
    """Dịch chuyển hình ảnh với một khoảng cách nhất định."""
    width, height = image.size
    shifted_image = Image.new("RGB", (width, height))
    shifted_image.paste(image, (shift[0], shift[1]))
    return shifted_image
def augment_image(image_data):
    # Nếu image_data là một FileStorage object từ Flask hoặc BytesIO, thì chuyển đổi thành đối tượng Image
    if hasattr(image_data, 'read'):  # Kiểm tra xem có phương thức 'read' không
        image = Image.open(image_data)
    elif isinstance(image_data, str):  # Nếu là một đường dẫn, mở đường dẫn đó
        image = Image.open(image_data)
    else:
        image = image_data  # Giả định rằng nếu không phải FileStorage, BytesIO, hoặc str, thì image_data đã là một đối tượng Image

    augmented_images = []

    # Xoay ảnh ±5 độ (giảm góc xoay để tránh biến dạng khuôn mặt)
    rotated = image.rotate(5, resample=Image.BICUBIC)
    augmented_images.append(rotated)
    rotated = image.rotate(-5, resample=Image.BICUBIC)
    augmented_images.append(rotated)

    # Lật ảnh ngang (giữ lại vì có thể hữu ích)
    flipped = ImageOps.mirror(image)
    augmented_images.append(flipped)

    # Giảm độ sáng đi 20%
    enhancer = ImageEnhance.Brightness(image)
    darkened = enhancer.enhance(0.8)  # Giảm sáng 20%
    augmented_images.append(darkened)

    # Dịch chuyển hình ảnh (giảm số lượng dịch chuyển)
    shifts = [(3, 0), (-3, 0)]  # Dịch chuyển nhẹ hơn
    for shift in shifts:
        shifted = shift_image(image, shift)
        augmented_images.append(shifted)

    # Thêm ảnh gốc vào danh sách
    augmented_images.append(image)

    return augmented_images

# def augment_image(image_data):
#     # Nếu image_data là một FileStorage object từ Flask hoặc BytesIO, thì chuyển đổi thành đối tượng Image
#     if hasattr(image_data, 'read'):  # Kiểm tra xem có phương thức 'read' không
#         image = Image.open(image_data)
#     elif isinstance(image_data, str):  # Nếu là một đường dẫn, mở đường dẫn đó
#         image = Image.open(image_data)
#     else:
#         image = image_data  # Giả định rằng nếu không phải FileStorage, BytesIO, hoặc str, thì image_data đã là một đối tượng Image

#     augmented_images = []

#     # Xoay ảnh ±10 độ
#     for angle in [-10, 10]:
#         rotated = image.rotate(angle, resample=Image.BICUBIC)
#         augmented_images.append(rotated)

#     # Lật ảnh ngang
#     flipped = ImageOps.mirror(image)
#     augmented_images.append(flipped)

#     # Điều chỉnh độ sáng
#     enhancer = ImageEnhance.Brightness(image)
#     for factor in [0.8, 1.2]:  # Giảm độ sáng 20% và tăng 20%
#         brightened = enhancer.enhance(factor)
#         augmented_images.append(brightened)

#     # Dịch chuyển hình ảnh theo các hướng khác nhau
#     shifts = [(5, 0), (-5, 0), (0, 5), (0, -5)]
#     for shift in shifts:
#         shifted = shift_image(image, shift)
#         augmented_images.append(shifted)

#     # Thêm ảnh gốc vào danh sách
#     augmented_images.append(image)

#     return augmented_images
