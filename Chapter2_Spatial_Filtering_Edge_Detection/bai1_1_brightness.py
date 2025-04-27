import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh từ file
# Bước 2: Định nghĩa giá trị thay đổi độ sáng (dương: tăng, âm: giảm)
# Bước 3: Cộng giá trị thay đổi vào mỗi pixel ảnh
# Bước 4: Đảm bảo giá trị pixel nằm trong khoảng [0, 255]
# Bước 5: Hiển thị ảnh gốc và ảnh đã thay đổi độ sáng
# Bước 6: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh (đọc ảnh màu)
image_path = 'image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    # Bước 2: Định nghĩa giá trị thay đổi độ sáng
    brightness_value_increase = 50  # Tăng sáng
    brightness_value_decrease = -50 # Giảm sáng

    print(f"Đang tăng độ sáng lên {brightness_value_increase} đơn vị...")
    print(f"Đang giảm độ sáng đi {-brightness_value_decrease} đơn vị...")

    # Bước 3 & 4: Thay đổi độ sáng và giới hạn giá trị pixel
    # Chuyển sang kiểu dữ liệu lớn hơn (int16) để tránh tràn số khi cộng/trừ
    # Sau đó dùng np.clip để giới hạn giá trị trong [0, 255]
    # Cuối cùng chuyển về uint8

    img_brighter = np.clip(img.astype(np.int16) + brightness_value_increase, 0, 255).astype(np.uint8)
    img_darker = np.clip(img.astype(np.int16) + brightness_value_decrease, 0, 255).astype(np.uint8)

    # Cách khác dùng hàm OpenCV (chỉ dùng cho tăng sáng với giá trị dương)
    # M_increase = np.ones(img.shape, dtype="uint8") * brightness_value_increase
    # img_brighter_cv = cv2.add(img, M_increase)

    # Bước 5: Hiển thị ảnh
    cv2.imshow('Anh Goc', img)
    cv2.imshow(f'Anh Sang Hon (+{brightness_value_increase})', img_brighter)
    cv2.imshow(f'Anh Toi Hon ({brightness_value_decrease})', img_darker)

    # Bước 6: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()