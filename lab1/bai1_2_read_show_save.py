import cv2
import numpy as np
import os # Thư viện os để kiểm tra file

# --- Các bước thực hiện ---
# Bước 1: Định nghĩa đường dẫn đến file ảnh.
# Bước 2: Đọc ảnh từ file bằng cv2.imread().
# Bước 3: Kiểm tra xem ảnh có được đọc thành công không.
# Bước 4: Hiển thị ảnh gốc lên màn hình bằng cv2.imshow().
# Bước 5: Chờ người dùng nhấn phím bất kỳ.
# Bước 6: Lưu ảnh đã đọc (có thể là ảnh gốc hoặc đã xử lý) với định dạng khác (ví dụ: PNG).
# Bước 7: Đóng tất cả các cửa sổ hiển thị ảnh.

# Bước 1: Đường dẫn ảnh
image_path = 'image.jpg'
output_path_png = 'output_image.png'
output_path_bmp = 'output_image.bmp'

print(f"Đang đọc ảnh từ: {image_path}")

# Bước 2: Đọc ảnh (mặc định là ảnh màu BGR)
img = cv2.imread(image_path)

# Bước 3: Kiểm tra đọc ảnh
if img is None:
    print(f"Lỗi: Không thể đọc được file ảnh tại '{image_path}'.")
    print("Hãy đảm bảo file tồn tại và OpenCV có quyền đọc.")
else:
    print("Đọc ảnh thành công!")
    height, width, channels = img.shape
    print(f"Kích thước ảnh: Rộng={width}, Cao={height}, Kênh={channels}")

    # Bước 4: Hiển thị ảnh
    window_name = 'Anh Goc (Nhan phim bat ky de dong)'
    cv2.imshow(window_name, img)
    print(f"Đã hiển thị ảnh trong cửa sổ '{window_name}'.")

    # Bước 5: Chờ nhấn phím
    print("Chờ người dùng nhấn phím...")
    cv2.waitKey(0) # Chờ vô hạn cho đến khi có phím được nhấn

    # Bước 6: Lưu ảnh với định dạng khác
    try:
        save_success_png = cv2.imwrite(output_path_png, img)
        save_success_bmp = cv2.imwrite(output_path_bmp, img)
        if save_success_png:
            print(f"Đã lưu ảnh thành công dưới dạng PNG tại: {output_path_png}")
        else:
            print(f"Lỗi khi lưu ảnh dạng PNG.")
        if save_success_bmp:
            print(f"Đã lưu ảnh thành công dưới dạng BMP tại: {output_path_bmp}")
        else:
            print(f"Lỗi khi lưu ảnh dạng BMP.")
    except Exception as e:
        print(f"Lỗi không xác định khi lưu ảnh: {e}")

    # Bước 7: Đóng cửa sổ
    cv2.destroyAllWindows()
    print("Đã đóng tất cả cửa sổ.")