import cv2
import numpy as np

# --- Các bước thực hiện (minh họa ý tưởng) ---
# Bước 1: Đọc ảnh và lấy ảnh cạnh Canny
# Bước 2: (Ý tưởng) Sử dụng ảnh cạnh cho các bước tiếp theo
#       - Tìm đường viền (Contours) từ ảnh cạnh
#       - Dùng đường viền để phân đoạn, đo đạc hoặc nhận dạng
#       - Khớp các mẫu cạnh (Edge template matching)

# Bước 1: Lấy ảnh cạnh Canny
image_path = 'image.jpg'
img = cv2.imread(image_path)
if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
    exit()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img_gray, 100, 200)

print("Đã phát hiện cạnh bằng Canny.")
cv2.imshow('Canny Edges', edges)

# Bước 2: (Ý tưởng) Sử dụng ảnh cạnh - Ví dụ tìm contours
print("Tìm các đường viền (contours) từ ảnh cạnh Canny...")
# cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]])
# - image: Ảnh nguồn, nhị phân (ảnh cạnh Canny là phù hợp). Ảnh sẽ bị thay đổi bởi hàm này.
# - mode: Chế độ lấy contour (vd: cv2.RETR_EXTERNAL chỉ lấy viền ngoài cùng, cv2.RETR_LIST lấy tất cả)
# - method: Phương pháp xấp xỉ contour (vd: cv2.CHAIN_APPROX_SIMPLE loại bỏ điểm thừa trên đường thẳng)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"- Tìm thấy {len(contours)} đường viền.")

# Vẽ các contours tìm được lên ảnh gốc để minh họa
img_contours = img.copy() # Tạo bản sao để vẽ lên
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2) # Vẽ tất cả contours (-1) màu xanh lá, dày 2 pixel

cv2.imshow('Anh voi Contours tim duoc', img_contours)

print("\nÝ tưởng kết hợp khác:")
print("- Phân đoạn ảnh: Dùng contours làm ranh giới các vùng.")
print("- Nhận dạng hình dạng: Phân tích hình học của contours (diện tích, chu vi, độ lồi...).")
print("- Khớp mẫu: So sánh các đoạn cạnh tìm được với các mẫu cạnh của đối tượng cần tìm.")


print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
cv2.waitKey(0)
cv2.destroyAllWindows()