import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh từ file (hoặc tạo ảnh đen nếu muốn vẽ trên nền trống)
# Bước 2: Vẽ đường thẳng bằng cv2.line()
# Bước 3: Vẽ hình chữ nhật bằng cv2.rectangle()
# Bước 4: Vẽ hình tròn bằng cv2.circle()
# Bước 5: Thêm văn bản bằng cv2.putText()
# Bước 6: Hiển thị ảnh đã vẽ
# Bước 7: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh
image_path = 'image.jpg'
img = cv2.imread(image_path)

# Hoặc tạo ảnh đen:
# height, width = 400, 600
# img = np.zeros((height, width, 3), dtype=np.uint8)

if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    print("Đang vẽ các hình cơ bản và thêm văn bản...")
    # --- Định nghĩa màu sắc (BGR) ---
    color_blue = (255, 0, 0)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_white = (255, 255, 255)

    # --- Định nghĩa độ dày đường vẽ ---
    thickness_line = 2
    thickness_filled = -1 # Giá trị -1 để tô đầy hình

    # Bước 2: Vẽ đường thẳng
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    start_point_line = (50, 50)
    end_point_line = (250, 150)
    cv2.line(img, start_point_line, end_point_line, color_red, thickness_line)
    print(f"- Đã vẽ đường thẳng từ {start_point_line} đến {end_point_line}")

    # Bước 3: Vẽ hình chữ nhật
    # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    # pt1: góc trên bên trái, pt2: góc dưới bên phải
    top_left_rect = (300, 50)
    bottom_right_rect = (500, 200)
    cv2.rectangle(img, top_left_rect, bottom_right_rect, color_green, thickness_line)
    print(f"- Đã vẽ hình chữ nhật từ {top_left_rect} đến {bottom_right_rect}")

    # Bước 4: Vẽ hình tròn
    # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
    center_circle = (150, 300)
    radius_circle = 50
    cv2.circle(img, center_circle, radius_circle, color_blue, thickness_filled) # Hình tròn tô đầy màu xanh dương
    print(f"- Đã vẽ hình tròn tâm {center_circle}, bán kính {radius_circle}")

    # Bước 5: Thêm văn bản
    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    text_to_add = "Xu ly anh - OpenCV"
    org_text = (50, 400) # Tọa độ góc dưới bên trái của văn bản
    font_face = cv2.FONT_HERSHEY_SIMPLEX # Chọn font chữ
    font_scale = 1 # Kích thước font
    text_thickness = 2
    cv2.putText(img, text_to_add, org_text, font_face, font_scale, color_white, text_thickness, cv2.LINE_AA) # LINE_AA cho chữ mượt hơn
    print(f"- Đã thêm văn bản '{text_to_add}' tại {org_text}")

    # Bước 6: Hiển thị ảnh đã vẽ
    cv2.imshow('Anh voi hinh ve va van ban', img)

    # Bước 7: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()