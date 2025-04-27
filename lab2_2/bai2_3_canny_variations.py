import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Các bước thực hiện ---
# Bước 1: Đọc ảnh gốc
# Bước 2: Tạo ra các phiên bản ảnh khác nhau (ví dụ: thêm nhiễu, giảm tương phản)
# Bước 3: Áp dụng Canny (dùng OpenCV cho đơn giản) lên các phiên bản ảnh đó
# Bước 4: Hiển thị các kết quả để so sánh và đánh giá

# --- Hàm thêm nhiễu Gaussian ---
def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255) # Giới hạn giá trị trong khoảng 0-255
    return noisy.astype(np.uint8)

# --- Hàm giảm tương phản ---
def decrease_contrast(image, alpha=0.7):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

# Bước 1: Đọc ảnh gốc
image_path = 'image.jpg'
img_original = cv2.imread(image_path)
if img_original is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
    exit()

img_gray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

# Bước 2: Tạo các phiên bản ảnh
print("Tạo các phiên bản ảnh khác nhau...")
img_noisy = add_gaussian_noise(img_original, sigma=30) # Thêm nhiễu nhiều hơn
img_gray_noisy = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)

img_low_contrast = decrease_contrast(img_original, alpha=0.5) # Giảm tương phản mạnh
img_gray_low_contrast = cv2.cvtColor(img_low_contrast, cv2.COLOR_BGR2GRAY)

# Bước 3: Áp dụng Canny cho các ảnh
print("Áp dụng Canny cho các ảnh...")
thresh1, thresh2 = 100, 200 # Sử dụng cùng ngưỡng để so sánh
edges_original = cv2.Canny(img_gray_original, thresh1, thresh2)
edges_noisy = cv2.Canny(img_gray_noisy, thresh1, thresh2) # Có thể cần làm mờ trước khi áp dụng Canny cho ảnh nhiễu
# edges_noisy_blurred = cv2.Canny(cv2.GaussianBlur(img_gray_noisy, (5,5), 0), thresh1, thresh2)
edges_low_contrast = cv2.Canny(img_gray_low_contrast, thresh1, thresh2) # Ngưỡng này có thể không tốt cho ảnh tương phản thấp

# Bước 4: Hiển thị kết quả
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.title('Ảnh Gốc')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(img_noisy, cv2.COLOR_BGR2RGB))
plt.title('Ảnh Nhiễu Gaussian')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(img_low_contrast, cv2.COLOR_BGR2RGB))
plt.title('Ảnh Tương phản thấp')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(edges_original, cmap='gray')
plt.title('Canny - Gốc')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(edges_noisy, cmap='gray')
plt.title('Canny - Nhiễu')
# plt.title('Canny - Nhiễu (có làm mờ)') # Nếu dùng ảnh đã làm mờ
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(edges_low_contrast, cmap='gray')
plt.title('Canny - Tương phản thấp')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nĐánh giá và Kết luận (quan sát ảnh kết quả):")
print("- Ảnh nhiễu: Canny phát hiện nhiều cạnh nhiễu. Việc làm mờ ảnh nhiễu trước khi dùng Canny có thể cải thiện.")
print("- Ảnh tương phản thấp: Canny có thể bỏ sót các cạnh yếu nếu ngưỡng không phù hợp. Cần điều chỉnh ngưỡng thấp hơn hoặc tiền xử lý tăng tương phản.")
print("- Canny hoạt động tốt nhất trên ảnh ít nhiễu và có độ tương phản tốt.")