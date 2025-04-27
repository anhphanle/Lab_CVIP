import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh từ file
# Bước 2: Định nghĩa hệ số thay đổi độ tương phản (alpha) và độ sáng (beta)
# Bước 3: Áp dụng công thức: new_pixel = alpha * old_pixel + beta
# Bước 4: Đảm bảo giá trị pixel nằm trong khoảng [0, 255] (cv2.convertScaleAbs tự xử lý)
# Bước 5: Hiển thị ảnh gốc và ảnh đã thay đổi độ tương phản
# Bước 6: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh
image_path = 'image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    # Bước 2: Định nghĩa hệ số alpha (tương phản) và beta (độ sáng)
    alpha_increase = 1.5  # Tăng tương phản (alpha > 1)
    alpha_decrease = 0.7  # Giảm tương phản (0 < alpha < 1)
    beta = 0             # Không thay đổi độ sáng kèm theo

    print(f"Đang tăng độ tương phản với alpha = {alpha_increase}")
    print(f"Đang giảm độ tương phản với alpha = {alpha_decrease}")

    # Bước 3 & 4: Áp dụng thay đổi tương phản bằng cv2.convertScaleAbs
    # Hàm này tự động nhân với alpha, cộng beta và giới hạn về [0, 255]
    img_contrast_increase = cv2.convertScaleAbs(img, alpha=alpha_increase, beta=beta)
    img_contrast_decrease = cv2.convertScaleAbs(img, alpha=alpha_decrease, beta=beta)

    # Bước 5: Hiển thị ảnh
    cv2.imshow('Anh Goc', img)
    cv2.imshow(f'Tuong Phan Tang (Alpha={alpha_increase})', img_contrast_increase)
    cv2.imshow(f'Tuong Phan Giam (Alpha={alpha_decrease})', img_contrast_decrease)

    # Bước 6: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()