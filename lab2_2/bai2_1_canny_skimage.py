import cv2
import numpy as np
from skimage import feature # Import module feature từ scikit-image
from skimage import io      # Có thể dùng io của skimage để đọc ảnh
import matplotlib.pyplot as plt # Dùng matplotlib để hiển thị cho nhất quán với skimage

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh và chuyển sang ảnh xám (nên dùng ảnh float [0,1] với skimage)
# Bước 2: Định nghĩa các tham số: sigma, ngưỡng dưới, ngưỡng trên
# Bước 3: Áp dụng hàm skimage.feature.canny
# Bước 4: Hiển thị ảnh gốc và ảnh cạnh Canny bằng Matplotlib
# Bước 5: Hiển thị cửa sổ Matplotlib

# Bước 1: Đọc ảnh và chuyển sang xám, chuẩn hóa về float [0, 1]
image_path = 'image.jpg'
try:
    # Đọc ảnh xám trực tiếp và chuẩn hóa
    img_gray_float = io.imread(image_path, as_gray=True)
    # Hoặc dùng OpenCV rồi chuyển đổi
    # img_cv = cv2.imread(image_path, 0) # Đọc xám bằng OpenCV
    # if img_cv is None: raise FileNotFoundError
    # img_gray_float = img_cv.astype(np.float32) / 255.0
except FileNotFoundError:
     print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
     exit()


# Bước 2: Định nghĩa tham số
sigma = 1.0           # Độ lệch chuẩn của bộ lọc Gaussian (làm mờ trước khi tính gradient)
low_threshold = 0.1   # Ngưỡng dưới (tương đối so với giá trị gradient lớn nhất)
high_threshold = 0.2  # Ngưỡng trên (tương đối)

print(f"Đang phát hiện cạnh bằng skimage.feature.canny...")
print(f"Tham số: sigma={sigma}, low_threshold={low_threshold}, high_threshold={high_threshold}")

# Bước 3: Áp dụng Canny của Scikit-image
# feature.canny(image, sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False)
# Ngưỡng trong skimage thường được tính tự động nếu không chỉ định, hoặc là tỉ lệ phần trăm của max gradient
edges_sk = feature.canny(img_gray_float, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)

# Kết quả là ảnh boolean (True/False), chuyển sang uint8 (0/255) để hiển thị giống OpenCV
edges_sk_display = (edges_sk * 255).astype(np.uint8)

# Bước 4: Hiển thị bằng Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img_gray_float, cmap='gray')
ax[0].set_title('Ảnh Xám Gốc')
ax[0].axis('off')

ax[1].imshow(edges_sk_display, cmap='gray')
ax[1].set_title('Canny Edges (Scikit-image)')
ax[1].axis('off')

plt.tight_layout()
# Bước 5: Hiển thị cửa sổ
plt.show()