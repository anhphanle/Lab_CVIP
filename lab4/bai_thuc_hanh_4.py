import cv2
import pywt
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from itertools import combinations # Để tạo các cặp ảnh

# --- Hàm Băm Wavelet ---
def wavelet_hash(image_path, wavelet='db4', level=1, hash_size=(8, 8)):
    """
    Tính toán mã băm wavelet cho một hình ảnh.

    Args:
        image_path (str): Đường dẫn đến file ảnh.
        wavelet (str): Loại wavelet sử dụng (ví dụ: 'haar', 'db4').
        level (int): Mức độ phân rã wavelet.
        hash_size (tuple): Kích thước mong muốn của ảnh xấp xỉ (LL) sau khi resize
                           trước khi tính toán median (ảnh hưởng đến độ dài hash).

    Returns:
        numpy.ndarray: Một mảng các bit (0 hoặc 1) đại diện cho mã băm,
                       hoặc None nếu không thể đọc ảnh.
    """
    try:
        # Đọc ảnh dưới dạng grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Cảnh báo: Không thể đọc ảnh {image_path}")
            return None

        # 1. Biến đổi Wavelet: Phân rã ảnh
        coeffs = pywt.wavedec2(img, wavelet, level=level)
        # Chỉ lấy hệ số xấp xỉ ở mức cao nhất (LL component)
        cA = coeffs[0]

        # 2. Giảm kích thước hệ số LL về kích thước cố định (hash_size)
        # Điều này giúp đảm bảo mã băm có độ dài nhất quán
        cA_resized = cv2.resize(cA, hash_size, interpolation=cv2.INTER_AREA)

        # 3. Lượng tử hóa hệ số: Tính toán median và tạo bit
        median_val = np.median(cA_resized)
        # Nếu giá trị > median -> 1, ngược lại -> 0
        hash_bits = (cA_resized > median_val).astype(int).flatten()

        return hash_bits

    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
        return None

# --- Hàm tính khoảng cách Hamming ---
def hamming_distance(hash1, hash2):
    """
    Tính khoảng cách Hamming giữa hai mã băm (phải cùng độ dài).

    Args:
        hash1 (numpy.ndarray): Mã băm thứ nhất.
        hash2 (numpy.ndarray): Mã băm thứ hai.

    Returns:
        int: Khoảng cách Hamming (số lượng bit khác nhau),
             hoặc -1 nếu hash không hợp lệ hoặc khác độ dài.
    """
    if hash1 is None or hash2 is None or len(hash1) != len(hash2):
        return -1 # Giá trị không hợp lệ
    return np.sum(hash1 != hash2) # Đếm số bit khác nhau

# --- Hàm đánh giá kết quả ---
def evaluate_similarity(results, threshold):
    """
    Đánh giá độ chính xác, độ nhạy, độ đặc hiệu dựa trên ngưỡng khoảng cách Hamming.

    Args:
        results (list): Danh sách các tuple (img1, img2, true_label, distance).
                        true_label=1 nếu tương tự, 0 nếu không.
        threshold (float): Ngưỡng khoảng cách Hamming để phân loại là tương tự.
                           distance <= threshold -> dự đoán là tương tự (1)
                           distance > threshold  -> dự đoán là không tương tự (0)

    Returns:
        dict: Dictionary chứa các chỉ số đánh giá (accuracy, precision, recall, specificity, conf_matrix).
              Trả về None nếu không có kết quả hợp lệ.
    """
    y_true = []
    y_pred = []
    valid_distances = [] # Dùng cho ROC

    has_valid_results = False
    for _, _, label, dist in results:
        if dist != -1: # Chỉ xét các cặp có khoảng cách hợp lệ
            y_true.append(label)
            prediction = 1 if dist <= threshold else 0
            y_pred.append(prediction)
            valid_distances.append(dist) # Lưu khoảng cách để vẽ ROC
            has_valid_results = True

    if not has_valid_results:
        print("Không có kết quả hợp lệ để đánh giá.")
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid_distances = np.array(valid_distances)

    if len(y_true) == 0:
        print("Không có nhãn đúng hợp lệ để đánh giá.")
        return None

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0) # Precision for "similar" class (1)
    recall = recall_score(y_true, y_pred, zero_division=0)       # Recall (Sensitivity) for "similar" class (1)

    # Tính Confusion Matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Độ đặc hiệu (Specificity) = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Tính ROC curve và AUC
    # Cần điểm số (score) thay vì dự đoán cứng. Dùng khoảng cách Hamming:
    # Điểm số cao hơn -> ít tương đồng hơn.
    # Để dùng roc_curve (mong muốn điểm số cao hơn = positive hơn),
    # ta có thể dùng (1 - normalized_distance) hoặc dùng distance và đảo ngược nhãn?
    # Cách đơn giản: coi distance là 'score' và roc_curve sẽ tự xử lý.
    # Hoặc chuẩn hóa: max_dist = len(results[0][3]) if results and results[0][3] is not None else 1
    # scores = 1 - (valid_distances / max_dist) # Score cao = tương tự
    # fpr, tpr, thresholds_roc = roc_curve(y_true, scores)

    # Dùng trực tiếp distance, roc_curve sẽ xử lý
    fpr, tpr, thresholds_roc = roc_curve(y_true, valid_distances, pos_label=0) # Xem xét "không tương tự" (0) là lớp positive để đường cong đi đúng hướng khi score là distance
    roc_auc = auc(fpr, tpr)


    evaluation = {
        'accuracy': accuracy,
        'precision': precision, # Độ chính xác cho lớp 'tương tự'
        'recall': recall,       # Độ nhạy (tìm được bao nhiêu % 'tương tự')
        'specificity': specificity, # Độ đặc hiệu (bao nhiêu % 'không tương tự' bị phân loại đúng)
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'roc_fpr': fpr,
        'roc_tpr': tpr,
        'roc_auc': roc_auc
    }
    return evaluation


# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":

    data_dir = "image" # Thư mục chứa ảnh

    # Liệt kê các file ảnh
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files.sort() # Sắp xếp để thứ tự nhất quán

    if not image_files:
        print(f"Lỗi: Không tìm thấy file ảnh nào trong thư mục '{data_dir}'.")
        exit()

    print(f"Tìm thấy {len(image_files)} ảnh:")
    for f in image_files: print(f"- {os.path.basename(f)}")

    image_pairs = [
        # Cặp Tương tự (Nhãn = 1)
        (os.path.join(data_dir, 'meo_goc.jpg'), os.path.join(data_dir, 'meo_goc_copy.jpg'), 1),
        (os.path.join(data_dir, 'meo_goc.jpg'), os.path.join(data_dir, 'meo_goc_nhieu.jpg'), 1),
        (os.path.join(data_dir, 'meo_goc.jpg'), os.path.join(data_dir, 'meo_goc_sang.jpg'), 1),

        # Cặp Không Tương tự (Nhãn = 0)
        (os.path.join(data_dir, 'meo_goc.jpg'), os.path.join(data_dir, 'cho_1.jpg'), 0),
        (os.path.join(data_dir, 'meo_goc.jpg'), os.path.join(data_dir, 'cay_xanh.png'), 0),
        (os.path.join(data_dir, 'meo_goc_nhieu.jpg'), os.path.join(data_dir, 'oto_do.jpg'), 0),
        (os.path.join(data_dir, 'cho_1.jpg'), os.path.join(data_dir, 'cay_xanh.png'), 0),
    ]

    if not image_pairs:
        print("\nLỗi: Danh sách 'image_pairs' rỗng. Vui lòng thêm dữ liệu vào code.")
        exit()

    # --- Tham số Băm và Đánh giá ---
    wavelet_type = 'db4'  # Loại wavelet
    decomp_level = 2     # Mức phân rã
    hash_dimension = (16, 16) # Kích thước ảnh LL để tạo hash (ảnh hưởng độ dài hash)
    hamming_threshold = 5 # Ngưỡng khoảng cách để coi là tương tự (cần tinh chỉnh)

    print(f"\nTham số Wavelet Hash: Wavelet={wavelet_type}, Level={decomp_level}, HashDim={hash_dimension}")
    print(f"Ngưỡng Hamming để phân loại 'Tương tự': {hamming_threshold}")

    # --- 1. Tạo mã băm cho tất cả ảnh ---
    print("\nĐang tạo mã băm cho các ảnh...")
    image_hashes = {}
    valid_image_files = [] # Chỉ lưu những file đọc và hash thành công
    for img_path in image_files:
        if not os.path.exists(img_path):
            print(f"Cảnh báo: File không tồn tại {img_path}")
            continue
        img_hash = wavelet_hash(img_path, wavelet=wavelet_type, level=decomp_level, hash_size=hash_dimension)
        if img_hash is not None:
            image_hashes[img_path] = img_hash
            valid_image_files.append(img_path)
            print(f"- Đã tạo hash cho {os.path.basename(img_path)} (Độ dài: {len(img_hash)})")
        else:
             print(f"- Bỏ qua {os.path.basename(img_path)} do lỗi hash.")

    if not image_hashes:
        print("\nLỗi: Không thể tạo mã băm cho bất kỳ ảnh nào.")
        exit()

    # --- 2. So sánh các cặp ảnh và tính khoảng cách Hamming ---
    print("\nĐang so sánh các cặp ảnh...")
    results = []
    for img1_path, img2_path, true_label in image_pairs:
        # Chỉ so sánh nếu cả hai ảnh đều có hash hợp lệ
        if img1_path in image_hashes and img2_path in image_hashes:
            hash1 = image_hashes[img1_path]
            hash2 = image_hashes[img2_path]
            dist = hamming_distance(hash1, hash2)
            results.append((os.path.basename(img1_path), os.path.basename(img2_path), true_label, dist))
            similarity_status = "Tương tự" if true_label == 1 else "Không tương tự"
            if dist != -1:
                print(f"- Cặp ({os.path.basename(img1_path)}, {os.path.basename(img2_path)}): Nhãn={similarity_status}, Khoảng cách Hamming = {dist}")
            else:
                print(f"- Cặp ({os.path.basename(img1_path)}, {os.path.basename(img2_path)}): Lỗi tính khoảng cách (có thể do hash lỗi hoặc khác độ dài).")
        else:
            print(f"- Bỏ qua cặp ({os.path.basename(img1_path)}, {os.path.basename(img2_path)}) do thiếu mã băm.")


    # --- 3. Đánh giá kết quả ---
    print(f"\nĐang đánh giá kết quả với ngưỡng Hamming = {hamming_threshold}...")
    evaluation_metrics = evaluate_similarity(results, hamming_threshold)

    if evaluation_metrics:
        print("\n--- Kết quả Đánh giá ---")
        print(f"Độ chính xác (Accuracy): {evaluation_metrics['accuracy']:.4f}")
        print(f"Độ chính xác lớp 'Tương tự' (Precision): {evaluation_metrics['precision']:.4f}")
        print(f"Độ nhạy lớp 'Tương tự' (Recall/Sensitivity): {evaluation_metrics['recall']:.4f}")
        print(f"Độ đặc hiệu lớp 'Không tương tự' (Specificity): {evaluation_metrics['specificity']:.4f}")
        cm = evaluation_metrics['confusion_matrix']
        print("Confusion Matrix:")
        print(f"  TN (Đúng Không tương tự): {cm['tn']:<5}  FP (Sai Tương tự): {cm['fp']:<5}")
        print(f"  FN (Sai Không tương tự): {cm['fn']:<5}  TP (Đúng Tương tự): {cm['tp']:<5}")
        print(f"Diện tích dưới đường cong ROC (AUC): {evaluation_metrics['roc_auc']:.4f}")

        # --- 4. Vẽ đường cong ROC ---
        print("\nĐang vẽ đường cong ROC...")
        plt.figure(figsize=(8, 6))
        lw = 2
        plt.plot(evaluation_metrics['roc_fpr'], evaluation_metrics['roc_tpr'], color='darkorange',
                 lw=lw, label=f'ROC curve (area = {evaluation_metrics["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') # Đường tham chiếu 45 độ
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity/Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    else:
        print("\nKhông thể thực hiện đánh giá do thiếu kết quả hợp lệ.")

    print("\nHoàn thành.")