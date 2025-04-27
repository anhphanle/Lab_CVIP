import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh và chuyển sang ảnh xám (cắt ngưỡng thường áp dụng trên ảnh xám)
# Bước 2: Chọn giá trị ngưỡng
# Bước 3: Áp dụng hàm cắt ngưỡng cv2.threshold với loại THRESH_BINARY
# Bước 4: Hiển thị ảnh xám gốc và ảnh nhị phân sau cắt ngưỡng
# Bước 5: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh và chuyển sang xám
image_path = 'image.jpg'
img_gray = cv2.imread(image_path, 0) # Số 0 để đọc ảnh xám

if img_gray is None:
    print(f"Lỗi: Không thể đọc ảnh hoặc chuyển sang xám từ: {image_path}")
else:
    # Bước 2: Chọn giá trị ngưỡng
    threshold_value = 127
    max_value = 255 # Giá trị gán cho pixel vượt ngưỡng

    print(f"Đang cắt ngưỡng ảnh xám với ngưỡng = {threshold_value}...")

    # Bước 3: Áp dụng cắt ngưỡng nhị phân
    # ret là giá trị ngưỡng được sử dụng (hữu ích với Otsu), thresh_img là ảnh kết quả
    ret, img_thresholded = cv2.threshold(img_gray, threshold_value, max_value, cv2.THRESH_BINARY)

    # Bạn cũng có thể thử ngưỡng Otsu để tự động tìm ngưỡng tối ưu
    # ret_otsu, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(f"Ngưỡng Otsu tự động tìm được: {ret_otsu}")
    # cv2.imshow('Anh Nhi Phan (Otsu)', img_otsu)

    # Bước 4: Hiển thị ảnh
    cv2.imshow('Anh Xam Goc', img_gray)
    cv2.imshow(f'Anh Nhi Phan (Nguong={threshold_value})', img_thresholded)

    # Bước 5: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()