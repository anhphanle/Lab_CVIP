import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh và chuyển sang ảnh xám (Canny hoạt động trên ảnh xám)
# Bước 2: Định nghĩa ngưỡng dưới (threshold1) và ngưỡng trên (threshold2)
# Bước 3: Áp dụng hàm cv2.Canny
# Bước 4: Hiển thị ảnh gốc và ảnh cạnh Canny
# Bước 5: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh và chuyển sang xám
image_path = 'image.jpg'
img = cv2.imread(image_path)
if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
    exit()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Bước 2: Định nghĩa ngưỡng (Giá trị thường dùng là khoảng 100 và 200)
threshold1 = 100  # Ngưỡng dưới
threshold2 = 200  # Ngưỡng trên

print(f"Đang phát hiện cạnh bằng cv2.Canny với ngưỡng {threshold1} và {threshold2}...")

# Bước 3: Áp dụng Canny
# cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])
# apertureSize: Kích thước kernel Sobel dùng bên trong (mặc định là 3)
# L2gradient: Sử dụng công thức tính độ lớn gradient chính xác hơn (True) hay nhanh hơn (False - mặc định)
edges_cv = cv2.Canny(img_gray, threshold1, threshold2)

# Bước 4: Hiển thị ảnh
cv2.imshow('Anh Xam Goc', img_gray)
cv2.imshow('Canny Edges (OpenCV)', edges_cv)

# Bước 5: Chờ và đóng
print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
cv2.waitKey(0)
cv2.destroyAllWindows()