import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh từ file.
# Bước 2: Cắt một vùng ảnh sử dụng slicing của NumPy.
# Bước 3: Thay đổi kích thước ảnh theo kích thước cố định (chiều rộng, chiều cao cụ thể).
# Bước 4: Thay đổi kích thước ảnh theo tỷ lệ (ví dụ: 50% kích thước gốc).
# Bước 5: Hiển thị ảnh gốc, ảnh đã cắt, và các ảnh đã thay đổi kích thước.
# Bước 6: Chờ người dùng nhấn phím và đóng cửa sổ.

# Bước 1: Đọc ảnh
image_path = 'image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    height, width = img.shape[:2]
    print(f"Kích thước ảnh gốc: Rộng={width}, Cao={height}")

    # Bước 2: Cắt ảnh (Crop)
    # Chọn tọa độ: y_start:y_end, x_start:x_end
    y1, y2 = height // 4, 3 * height // 4 # Cắt từ 1/4 đến 3/4 chiều cao
    x1, x2 = width // 4, 3 * width // 4   # Cắt từ 1/4 đến 3/4 chiều rộng
    print(f"Đang cắt vùng ảnh từ y=[{y1}:{y2}), x=[{x1}:{x2})")
    img_cropped = img[y1:y2, x1:x2]

    # Bước 3: Thay đổi kích thước cố định (Resize to specific dimensions)
    new_width = 300
    new_height = 200
    print(f"Đang thay đổi kích thước thành: Rộng={new_width}, Cao={new_height}")
    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    # dsize: tuple (width, height)
    img_resized_fixed = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR) # INTER_LINEAR là phổ biến

    # Bước 4: Thay đổi kích thước theo tỷ lệ (Resize by scale factor)
    scale_percent = 50 # Giảm còn 50%
    scale_factor = scale_percent / 100.0
    print(f"Đang thay đổi kích thước theo tỷ lệ: {scale_percent}%")
    # Có thể dùng fx, fy hoặc tính dsize từ tỉ lệ
    # Cách 1: dùng fx, fy
    img_resized_scale = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA) # INTER_AREA tốt cho thu nhỏ
    # Cách 2: tính dsize
    # scaled_width = int(width * scale_factor)
    # scaled_height = int(height * scale_factor)
    # img_resized_scale = cv2.resize(img, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)


    # Bước 5: Hiển thị các ảnh
    cv2.imshow('Anh Goc', img)
    if img_cropped.size > 0: # Kiểm tra xem có cắt được không
        cv2.imshow('Anh Da Cat (Cropped)', img_cropped)
    cv2.imshow(f'Resize Co Dinh ({new_width}x{new_height})', img_resized_fixed)
    cv2.imshow(f'Resize Ty Le ({scale_percent}%)', img_resized_scale)

    # Bước 6: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()