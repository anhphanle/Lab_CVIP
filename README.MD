# Bài Thực Hành Môn Xử Lý Ảnh và Thị Giác Máy Tính

## Thông tin sinh viên

- **Họ và tên:** Phan Lê Nhật Anh
- **MSSV:** 22H4060068
- **Lớp:** CN21CLCA
- **Mã Học Phần:** [010412103601] - Xử lý ảnh và thị giác máy tính

## Giới thiệu

Repository này chứa tổng hợp các bài tập thực hành cho môn học Xử lý Ảnh và Thị Giác Máy Tính (Computer Vision and Image Processing). Các bài thực hành bao gồm việc triển khai và thử nghiệm các thuật toán xử lý ảnh cơ bản đến nâng cao bằng ngôn ngữ Python và các thư viện phổ biến như OpenCV, Matplotlib, Scikit-image, PyWavelets.

## Cấu trúc Repository

Repository được tổ chức theo từng bài thực hành hoặc chương học, với mỗi phần được đặt trong thư mục hoặc file riêng biệt.

- **/Chapter1_Basic_Operations:** Các bài thực hành về thao tác cơ bản (đọc/ghi/hiển thị ảnh, chuyển đổi không gian màu, cắt xén, thay đổi kích thước, vẽ hình).
- **/Chapter2_Spatial_Filtering_Edge_Detection:** Các bài thực hành về lọc không gian (lọc trung bình, Gaussian, trung vị, làm sắc nét) và phát hiện cạnh (Sobel, Prewitt, Laplacian, Canny).
- **/Chapter2_Fourier_Wavelet_Geometric_Morphology:** Các bài thực hành về biến đổi Fourier, Wavelet, biến đổi hình học và xử lý hình thái học.
- **/Lab4_Wavelet_Hashing:** Bài thực hành về so sánh sự tương đồng ảnh sử dụng băm Wavelet.
- `README.md`: File này, cung cấp thông tin tổng quan về repository.

## Các kỹ thuật chính được thực hành

- **Thao tác điểm ảnh:** Thay đổi độ sáng, tương phản, âm bản, phân ngưỡng.
- **Histogram:** Tính toán, vẽ và cân bằng histogram.
- **Lọc không gian:** Lọc trung bình, Gaussian, trung vị, làm sắc nét, lọc song phương (Bilateral).
- **Phát hiện cạnh:** Sobel, Prewitt, Laplacian, Canny.
- **Biến đổi Fourier:** Tính toán FFT, hiển thị phổ, lọc tần số cơ bản.
- **Biến đổi Wavelet:** Phân rã ảnh, băm wavelet (Wavelet Hashing).
- **Biến đổi hình học:** Thay đổi kích thước, cắt xén, quay, tịnh tiến, affine, perspective.
- **Xử lý hình thái học:** Co, giãn, mở, đóng, gradient, top-hat, black-hat.
- **(Nâng cao):** Phát hiện đặc trưng (SIFT, SURF, ORB), đối sánh ảnh.

## Hướng dẫn sử dụng

1.  **Clone repository:**
    ```bash
    git clone https://github.com/anhphanle/Lab_CVIP.git
    cd Lab_CVIP
    ```
2.  **Cài đặt thư viện:** Đảm bảo bạn đã cài đặt các thư viện cần thiết. Một file `requirements.txt` có thể được cung cấp để cài đặt dễ dàng hơn:
    ```bash
    pip install -r requirements.txt
    ```
    Hoặc cài đặt thủ công các thư viện chính:
    ```bash
    pip install opencv-python numpy matplotlib Pillow scikit-image PyWavelets scikit-learn
    ```
3.  **Chuẩn bị ảnh:** Đặt các ảnh cần thiết vào đúng thư mục (ví dụ: thư mục gốc hoặc thư mục `image_data`) theo yêu cầu của từng bài thực hành.
4.  **Chạy code:** Di chuyển vào thư mục chứa file bài thực hành cụ thể và chạy bằng lệnh:
    ```bash
    python ten_file_bai_thuc_hanh.py
    ```

## Liên kết

- **GitHub Repository:** [https://github.com/anhphanle/Lab_CVIP](https://github.com/anhphanle/Lab_CVIP)

---
