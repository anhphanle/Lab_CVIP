import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh từ file
# Bước 2: Áp dụng bộ lọc trung vị (Median Filter)
# Bước 3: Áp dụng bộ lọc song phương (Bilateral Filter)
# Bước 4: Hiển thị ảnh gốc và các ảnh đã lọc phi tuyến
# Bước 5: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh
image_path = 'image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    print("Đang áp dụng các bộ lọc phi tuyến...")
    # --- Tham số lọc ---
    ksize_median = 5 # Kích thước kernel Median (số lẻ)
    bilateral_d = 9    # Đường kính lân cận cho Bilateral
    bilateral_sigmaColor = 75 # Sigma màu
    bilateral_sigmaSpace = 75 # Sigma không gian

    # --- Bước 2: Lọc Trung vị ---
    # cv2.medianBlur(src, ksize[, dst])
    img_median = cv2.medianBlur(img, ksize_median)
    print(f"- Đã áp dụng lọc Median với kernel size = {ksize_median}")

    # --- Bước 3: Lọc Song phương ---
    # cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]])
    img_bilateral = cv2.bilateralFilter(img, d=bilateral_d, sigmaColor=bilateral_sigmaColor, sigmaSpace=bilateral_sigmaSpace)
    print(f"- Đã áp dụng lọc Bilateral với d={bilateral_d}, sigmaColor={bilateral_sigmaColor}, sigmaSpace={bilateral_sigmaSpace}")

    # --- Bước 4: Hiển thị ảnh ---
    cv2.imshow('Anh Goc', img)
    cv2.imshow(f'Loc Trung Vi (Median k={ksize_median})', img_median)
    cv2.imshow(f'Loc Song Phuong (Bilateral d={bilateral_d})', img_bilateral)

    # --- Bước 5: Chờ và đóng ---
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()