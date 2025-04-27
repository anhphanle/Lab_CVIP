import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh từ file
# Bước 2: Tự định nghĩa một kernel tùy chỉnh (ví dụ: làm nổi khối - emboss)
# Bước 3: Áp dụng tích chập với kernel tùy chỉnh bằng cv2.filter2D
# Bước 4: Hiển thị ảnh gốc và ảnh đã xử lý với kernel tùy chỉnh
# Bước 5: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh
image_path = 'image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    # Bước 2: Định nghĩa kernel tùy chỉnh (ví dụ: Emboss)
    # Kernel này tạo hiệu ứng làm nổi khối cho ảnh
    kernel_emboss = np.array([[-2, -1, 0],
                              [-1,  1, 1],
                              [ 0,  1, 2]], dtype=np.float32)
    print("Đang áp dụng bộ lọc Emboss với kernel tùy chỉnh...")
    print("Kernel được sử dụng:\n", kernel_emboss)

    # Bước 3: Áp dụng tích chập với kernel tùy chỉnh
    img_embossed = cv2.filter2D(img, -1, kernel_emboss)

    # Thường cộng thêm 128 để đưa các giá trị âm về dải nhìn thấy được
    img_embossed_display = cv2.add(img_embossed, 128)

    # Bước 4: Hiển thị ảnh
    cv2.imshow('Anh Goc', img)
    cv2.imshow('Anh Embossed', img_embossed_display)

    # Bước 5: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()