import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh từ file
# Bước 2: Định nghĩa kích thước kernel (mặt nạ) lọc trung bình (ví dụ: 5x5)
# Bước 3: Áp dụng hàm lọc trung bình cv2.blur
# Bước 4: Hiển thị ảnh gốc và ảnh đã làm mờ
# Bước 5: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh
image_path = 'image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    # Bước 2: Định nghĩa kích thước kernel (phải là số lẻ)
    kernel_size = (5, 5) # ví dụ kernel 5x5
    print(f"Đang áp dụng lọc trung bình với kernel {kernel_size}...")

    # Bước 3: Áp dụng lọc trung bình
    img_blurred = cv2.blur(img, kernel_size)

    # Bước 4: Hiển thị ảnh
    cv2.imshow('Anh Goc', img)
    cv2.imshow(f'Loc Trung Binh {kernel_size}', img_blurred)

    # Bước 5: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()