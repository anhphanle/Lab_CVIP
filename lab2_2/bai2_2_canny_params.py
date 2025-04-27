import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage import io

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh và chuyển sang ảnh xám
# Bước 2: Thực hiện Canny với các bộ tham số khác nhau (OpenCV và Scikit-image)
# Bước 3: Hiển thị các kết quả cạnh nhau để so sánh

# Bước 1: Đọc ảnh
image_path = 'image.jpg'
img_cv = cv2.imread(image_path, 0) # Đọc xám cho OpenCV
try:
    img_sk = io.imread(image_path, as_gray=True) # Đọc float [0,1] cho Scikit-image
except FileNotFoundError:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
    exit()

if img_cv is None:
    print("Lỗi đọc ảnh với OpenCV")
    exit()

print("Thực hiện Canny với các tham số khác nhau...")

# --- Bước 2: Thực hiện Canny với các tham số khác nhau ---

# OpenCV
thresh1_low, thresh2_low = 50, 100
thresh1_high, thresh2_high = 150, 250
edges_cv_low = cv2.Canny(img_cv, thresh1_low, thresh2_low)
edges_cv_high = cv2.Canny(img_cv, thresh1_high, thresh2_high)
edges_cv_default = cv2.Canny(img_cv, 100, 200) # Tham số "thông thường"

# Scikit-image
sigma_low, low_t_low, high_t_low = 0.5, 0.05, 0.15 # Ít mờ, ngưỡng thấp
sigma_high, low_t_high, high_t_high = 3.0, 0.1, 0.3 # Mờ nhiều, ngưỡng cao
edges_sk_low = (feature.canny(img_sk, sigma=sigma_low, low_threshold=low_t_low, high_threshold=high_t_low) * 255).astype(np.uint8)
edges_sk_high = (feature.canny(img_sk, sigma=sigma_high, low_threshold=low_t_high, high_threshold=high_t_high) * 255).astype(np.uint8)
edges_sk_default = (feature.canny(img_sk, sigma=1.0, low_threshold=0.1, high_threshold=0.2) * 255).astype(np.uint8)

# --- Bước 3: Hiển thị kết quả ---
plt.figure(figsize=(15, 10)) # Kích thước lớn hơn

# Hàng 1: OpenCV
plt.subplot(2, 4, 1)
plt.imshow(img_cv, cmap='gray')
plt.title('Ảnh Gốc (OpenCV)')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(edges_cv_low, cmap='gray')
plt.title(f'CV Canny ({thresh1_low},{thresh2_low})')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(edges_cv_default, cmap='gray')
plt.title('CV Canny (100,200)')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(edges_cv_high, cmap='gray')
plt.title(f'CV Canny ({thresh1_high},{thresh2_high})')
plt.axis('off')

# Hàng 2: Scikit-image
plt.subplot(2, 4, 5)
plt.imshow(img_sk, cmap='gray')
plt.title('Ảnh Gốc (Skimage)')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(edges_sk_low, cmap='gray')
plt.title(f'SK Canny s={sigma_low} ({low_t_low:.2f},{high_t_low:.2f})')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(edges_sk_default, cmap='gray')
plt.title('SK Canny s=1 (0.10,0.20)')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(edges_sk_high, cmap='gray')
plt.title(f'SK Canny s={sigma_high} ({low_t_high:.2f},{high_t_high:.2f})')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nQuan sát kết quả:")
print("- Ngưỡng thấp (cả 2 threshold) -> nhiều cạnh nhiễu hơn.")
print("- Ngưỡng cao -> ít cạnh nhiễu, nhưng có thể mất cạnh yếu.")
print("- Sigma cao (Skimage) -> ảnh mờ hơn trước khi tính cạnh -> cạnh ít chi tiết, dày hơn, ít nhiễu hơn.")
print("- Sigma thấp (Skimage) -> giữ nhiều chi tiết, nhạy cảm với nhiễu hơn.")