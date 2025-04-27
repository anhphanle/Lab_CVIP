import cv2
import numpy as np

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh và chuyển sang ảnh xám
# Bước 2: Áp dụng toán tử Sobel để tính gradient theo x và y
# Bước 3: Kết hợp gradient x, y để tính độ lớn gradient tổng (cạnh Sobel)
# Bước 4: Định nghĩa kernel Prewitt và áp dụng cv2.filter2D để tính gradient Prewitt
# Bước 5: Kết hợp gradient Prewitt x, y để tính độ lớn gradient tổng (cạnh Prewitt)
# Bước 6: Hiển thị ảnh gốc, các thành phần gradient và ảnh cạnh
# Bước 7: Chờ người dùng nhấn phím và đóng cửa sổ

# Bước 1: Đọc ảnh và chuyển sang xám
image_path = 'image.jpg'
img_gray = cv2.imread(image_path, 0)

if img_gray is None:
    print(f"Lỗi: Không thể đọc ảnh hoặc chuyển sang xám từ: {image_path}")
else:
    ksize = 3 # Kích thước kernel cho Sobel (thường là 3 hoặc 5)

    # --- SOBEL ---
    print(f"Đang tính toán cạnh bằng Sobel (ksize={ksize})...")
    # Bước 2: Tính Sobel x và y
    # Sử dụng cv2.CV_64F để tránh mất thông tin giá trị âm của đạo hàm
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize) # Đạo hàm theo x
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize) # Đạo hàm theo y

    # Bước 3: Tính độ lớn gradient Sobel
    # Cách 1: Dùng cv2.magnitude
    # sobel_magnitude = cv2.magnitude(sobelx, sobely)
    # Cách 2: Tính toán thủ công
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Chuyển đổi về uint8 để hiển thị
    sobelx_display = cv2.convertScaleAbs(sobelx)
    sobely_display = cv2.convertScaleAbs(sobely)
    sobel_magnitude_display = cv2.convertScaleAbs(sobel_magnitude)

    # --- PREWITT ---
    print("Đang tính toán cạnh bằng Prewitt...")
    # Bước 4: Định nghĩa kernel Prewitt và áp dụng filter2D
    kernel_prewitt_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]], dtype=np.float32)
    kernel_prewitt_y = np.array([[-1, -1, -1],
                                 [ 0,  0,  0],
                                 [ 1,  1,  1]], dtype=np.float32)

    prewittx = cv2.filter2D(img_gray.astype(np.float32), -1, kernel_prewitt_x) # Chuyển sang float để tính toán chính xác hơn
    prewitty = cv2.filter2D(img_gray.astype(np.float32), -1, kernel_prewitt_y)

    # Bước 5: Tính độ lớn gradient Prewitt
    prewitt_magnitude = np.sqrt(prewittx**2 + prewitty**2)

    # Chuyển đổi về uint8 để hiển thị
    prewittx_display = cv2.convertScaleAbs(prewittx)
    prewitty_display = cv2.convertScaleAbs(prewitty)
    prewitt_magnitude_display = cv2.convertScaleAbs(prewitt_magnitude)


    # Bước 6: Hiển thị kết quả
    cv2.imshow('Anh Xam Goc', img_gray)
    cv2.imshow('Sobel X', sobelx_display)
    cv2.imshow('Sobel Y', sobely_display)
    cv2.imshow('Sobel Magnitude', sobel_magnitude_display)
    # cv2.imshow('Prewitt X', prewittx_display)
    # cv2.imshow('Prewitt Y', prewitty_display)
    cv2.imshow('Prewitt Magnitude', prewitt_magnitude_display)


    # Bước 7: Chờ và đóng
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()