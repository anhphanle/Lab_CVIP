import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh từ file
# Bước 2: Định nghĩa kernel làm sắc nét (ví dụ: kernel dựa trên Laplacian)
# Bước 3: Áp dụng tích chập với kernel tùy chỉnh bằng cv2.filter2D
# Bước 4: Hiển thị ảnh gốc và ảnh đã làm sắc nét
# Bước 5: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh
image_path = 'image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    # Bước 2: Định nghĩa kernel làm sắc nét
    # Kernel 1: Tăng cường pixel trung tâm
    # kernel_sharpen = np.array([[ 0, -1,  0],
    #                            [-1,  5, -1],
    #                            [ 0, -1,  0]], dtype=np.float32)
    # Kernel 2: Một biến thể khác
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]], dtype=np.float32)
    print("Đang áp dụng bộ lọc làm sắc nét...")
    print("Kernel được sử dụng:\n", kernel_sharpen)

    # Bước 3: Áp dụng tích chập với kernel tùy chỉnh
    # cv2.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])
    # ddepth = -1 nghĩa là ảnh output có cùng độ sâu (số bit màu) với ảnh input
    img_sharpened = cv2.filter2D(img, -1, kernel_sharpen)

    # Bước 4: Hiển thị ảnh
    cv2.imshow('Anh Goc', img)
    cv2.imshow('Anh Lam Sac Net', img_sharpened)

    # Bước 5: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()