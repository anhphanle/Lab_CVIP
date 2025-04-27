import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh màu từ file.
# Bước 2: Chuyển đổi ảnh từ BGR (mặc định của OpenCV) sang Grayscale.
# Bước 3: Chuyển đổi ảnh từ BGR sang HSV.
# Bước 4: Chuyển đổi ảnh từ BGR sang LAB.
# Bước 5: Hiển thị ảnh gốc và các ảnh đã chuyển đổi không gian màu.
# Bước 6: Chờ người dùng nhấn phím và đóng cửa sổ.

# Bước 1: Đọc ảnh màu
image_path = 'image.jpg'
img_bgr = cv2.imread(image_path)

if img_bgr is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    print("Đang chuyển đổi không gian màu...")

    # Bước 2: BGR sang Grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print("- Đã chuyển sang Grayscale.")

    # Bước 3: BGR sang HSV (Hue, Saturation, Value)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    print("- Đã chuyển sang HSV.")

    # Bước 4: BGR sang LAB (L*a*b*)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    print("- Đã chuyển sang LAB.")

    # Bước 5: Hiển thị các ảnh
    cv2.imshow('Anh Goc (BGR)', img_bgr)
    cv2.imshow('Anh Grayscale', img_gray)
    cv2.imshow('Anh HSV', img_hsv) # Hiển thị trực tiếp có thể không như mong đợi về màu sắc
    cv2.imshow('Anh LAB', img_lab) # Hiển thị trực tiếp có thể không như mong đợi về màu sắc

    # Bước 6: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()