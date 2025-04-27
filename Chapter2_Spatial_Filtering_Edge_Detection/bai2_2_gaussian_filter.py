import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh từ file
# Bước 2: Định nghĩa kích thước kernel và độ lệch chuẩn sigmaX (hoặc sigmaY)
# Bước 3: Áp dụng hàm lọc Gaussian cv2.GaussianBlur
# Bước 4: Hiển thị ảnh gốc và ảnh đã làm mờ Gaussian
# Bước 5: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh
image_path = 'image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    # Bước 2: Định nghĩa kích thước kernel (phải là số lẻ) và sigma
    kernel_size = (5, 5) # ví dụ kernel 5x5
    sigmaX = 0 # Nếu sigmaX = 0, OpenCV sẽ tự tính dựa trên kernel_size
    print(f"Đang áp dụng lọc Gaussian với kernel {kernel_size} và sigmaX={sigmaX}...")

    # Bước 3: Áp dụng lọc Gaussian
    # cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
    img_gaussian_blurred = cv2.GaussianBlur(img, kernel_size, sigmaX=sigmaX)

    # Bước 4: Hiển thị ảnh
    cv2.imshow('Anh Goc', img)
    cv2.imshow(f'Loc Gaussian {kernel_size}', img_gaussian_blurred)

    # Bước 5: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()