import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh từ file
# Bước 2: Áp dụng công thức âm bản: negative_pixel = 255 - original_pixel
# Bước 3: Hiển thị ảnh gốc và ảnh âm bản
# Bước 4: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh (có thể đọc màu hoặc xám)
image_path = 'image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    print("Đang tạo ảnh âm bản...")
    # Bước 2: Tạo ảnh âm bản
    # Phép toán này áp dụng cho từng pixel nhờ khả năng của NumPy/OpenCV
    img_negative = 255 - img

    # Bước 3: Hiển thị ảnh
    cv2.imshow('Anh Goc', img)
    cv2.imshow('Anh Am Ban (Negative)', img_negative)

    # Bước 4: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()