import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh từ file
# Bước 2: Áp dụng các bộ lọc khác nhau (Mean, Gaussian, Median, Bilateral)
# Bước 3: Hiển thị ảnh gốc và tất cả các ảnh đã lọc để so sánh
# Bước 4: Chờ người dùng nhấn phím và đóng cửa sổ (hoặc dùng Matplotlib)

# Bước 1: Đọc ảnh
image_path = 'image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    print("Đang áp dụng các bộ lọc khác nhau để so sánh...")
    # --- Các tham số lọc ---
    ksize_5 = (5, 5)
    ksize_median = 5
    bilateral_d = 9
    bilateral_sigma = 75

    # --- Bước 2: Áp dụng các bộ lọc ---
    img_mean = cv2.blur(img, ksize_5)
    img_gaussian = cv2.GaussianBlur(img, ksize_5, sigmaX=0)
    img_median = cv2.medianBlur(img, ksize_median)
    img_bilateral = cv2.bilateralFilter(img, d=bilateral_d, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma)

    # --- Bước 3: Hiển thị bằng Matplotlib để dễ so sánh ---
    plt.figure(figsize=(12, 8)) # Kích thước cửa sổ đồ thị

    plt.subplot(2, 3, 1) # 2 hàng, 3 cột, vị trí 1
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Chuyển sang RGB cho Matplotlib
    plt.title('Ảnh Gốc')
    plt.axis('off') # Ẩn trục tọa độ

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img_mean, cv2.COLOR_BGR2RGB))
    plt.title(f'Mean Filter {ksize_5}')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(img_gaussian, cv2.COLOR_BGR2RGB))
    plt.title(f'Gaussian Filter {ksize_5}')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(img_median, cv2.COLOR_BGR2RGB))
    plt.title(f'Median Filter k={ksize_median}')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2RGB))
    plt.title(f'Bilateral d={bilateral_d}, s={bilateral_sigma}')
    plt.axis('off')

    plt.tight_layout() # Tự động điều chỉnh khoảng cách
    plt.show() # Hiển thị cửa sổ đồ thị

    # --- Hoặc hiển thị bằng OpenCV ---
    # cv2.imshow('Anh Goc', img)
    # cv2.imshow(f'Mean {ksize_5}', img_mean)
    # cv2.imshow(f'Gaussian {ksize_5}', img_gaussian)
    # cv2.imshow(f'Median k={ksize_median}', img_median)
    # cv2.imshow(f'Bilateral d={bilateral_d}', img_bilateral)
    #
    # print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()