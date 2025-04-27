import cv2
import numpy as np
import os

# --- Các hàm xử lý ảnh ---

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Thêm nhiễu Gaussian vào ảnh.

    Args:
        image (numpy.ndarray): Ảnh đầu vào (BGR hoặc Grayscale).
        mean (int): Giá trị trung bình của nhiễu (thường là 0).
        sigma (int): Độ lệch chuẩn của nhiễu (càng lớn càng nhiễu).

    Returns:
        numpy.ndarray: Ảnh đã được thêm nhiễu.
    """
    # Tạo nhiễu Gaussian có cùng kích thước với ảnh
    # image.shape trả về (height, width, channels) cho ảnh màu
    # hoặc (height, width) cho ảnh xám
    gauss = np.random.normal(mean, sigma, image.shape)

    # Cộng nhiễu vào ảnh (cần chuyển ảnh sang float để tránh lỗi tràn số)
    noisy_float = image.astype(np.float32) + gauss

    # Giới hạn giá trị pixel trong khoảng [0, 255]
    noisy_clipped = np.clip(noisy_float, 0, 255)

    # Chuyển ảnh về lại kiểu dữ liệu gốc (uint8)
    noisy_uint8 = noisy_clipped.astype(np.uint8)
    return noisy_uint8

def change_brightness(image, value):
    """
    Thay đổi độ sáng của ảnh bằng cách cộng/trừ một giá trị cố định.

    Args:
        image (numpy.ndarray): Ảnh đầu vào (BGR hoặc Grayscale).
        value (int): Giá trị cần cộng vào độ sáng (dương để tăng, âm để giảm).

    Returns:
        numpy.ndarray: Ảnh đã được thay đổi độ sáng.
    """
    # Chuyển sang kiểu int16 để tránh tràn số khi cộng/trừ
    img_int = image.astype(np.int16)
    img_bright_int = img_int + value

    # Giới hạn giá trị pixel trong khoảng [0, 255]
    img_bright_clipped = np.clip(img_bright_int, 0, 255)

    # Chuyển về lại kiểu dữ liệu gốc (uint8)
    img_bright_final = img_bright_clipped.astype(np.uint8)
    return img_bright_final

# --- Chương trình chính ---

if __name__ == "__main__":
    # 1. Định nghĩa đường dẫn và tham số
    input_image_path = 'image.jpg'  # !!! THAY ĐỔI thành đường dẫn ảnh của bạn !!!
    output_dir = 'modified_images'   # Thư mục để lưu ảnh kết quả
    noise_sigma = 30                 # Độ mạnh của nhiễu Gaussian
    brightness_increase = 60         # Mức độ tăng sáng

    # Tạo thư mục output nếu chưa có
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Đã tạo thư mục lưu ảnh: '{output_dir}'")
        except OSError as e:
            print(f"Lỗi: Không thể tạo thư mục '{output_dir}'. {e}")
            exit() # Thoát nếu không tạo được thư mục

    # 2. Đọc ảnh gốc
    img_original = cv2.imread(input_image_path)

    # Kiểm tra xem ảnh đã đọc thành công chưa
    if img_original is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: '{input_image_path}'")
        print("Hãy đảm bảo file ảnh tồn tại và đường dẫn chính xác.")
        exit() # Thoát chương trình
    else:
        print(f"Đã đọc ảnh thành công: '{input_image_path}'")

    # 3. Thêm nhiễu vào ảnh
    print(f"Đang thêm nhiễu Gaussian với sigma={noise_sigma}...")
    img_noisy = add_gaussian_noise(img_original, sigma=noise_sigma)
    print("-> Hoàn thành thêm nhiễu.")

    # 4. Tăng độ sáng ảnh
    print(f"Đang tăng độ sáng lên {brightness_increase} đơn vị...")
    img_brighter = change_brightness(img_original, value=brightness_increase)
    print("-> Hoàn thành tăng độ sáng.")

    # 5. Lưu các ảnh đã xử lý
    # Lấy tên file gốc không bao gồm đuôi
    base_filename = os.path.splitext(os.path.basename(input_image_path))[0]

    # Tạo tên file output
    noisy_output_path = os.path.join(output_dir, f"{base_filename}_noisy_s{noise_sigma}.jpg")
    brighter_output_path = os.path.join(output_dir, f"{base_filename}_brighter_p{brightness_increase}.jpg")

    try:
        # Lưu ảnh nhiễu
        success_noisy = cv2.imwrite(noisy_output_path, img_noisy)
        if success_noisy:
            print(f"Đã lưu ảnh nhiễu tại: '{noisy_output_path}'")
        else:
            print(f"Lỗi: Không thể lưu ảnh nhiễu tại '{noisy_output_path}'")

        # Lưu ảnh sáng hơn
        success_brighter = cv2.imwrite(brighter_output_path, img_brighter)
        if success_brighter:
            print(f"Đã lưu ảnh sáng hơn tại: '{brighter_output_path}'")
        else:
            print(f"Lỗi: Không thể lưu ảnh sáng hơn tại '{brighter_output_path}'")

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình lưu ảnh: {e}")

    # 6. (Tùy chọn) Hiển thị các ảnh để xem trước
    print("\nĐang hiển thị các ảnh (Nhấn phím bất kỳ trên cửa sổ ảnh để đóng)...")
    cv2.imshow('Anh Goc', img_original)
    cv2.imshow(f'Anh Nhieu (Sigma={noise_sigma})', img_noisy)
    cv2.imshow(f'Anh Sang Hon (+{brightness_increase})', img_brighter)

    cv2.waitKey(0) # Chờ người dùng nhấn phím
    cv2.destroyAllWindows() # Đóng tất cả cửa sổ
    print("Đã đóng cửa sổ hiển thị.")