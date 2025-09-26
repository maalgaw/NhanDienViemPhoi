# Chẩn đoán viêm phổi qua ảnh X-quang

![Made by Trần Quang Lâm](https://img.shields.io/badge/Made%20by%20Trần%20Quang%20Lâm-blue?style=for-the-badge)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-orange?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?style=for-the-badge&logo=scikit-learn)
![Kaggle Dataset](https://img.shields.io/badge/Kaggle%20Dataset-red?style=for-the-badge&logo=kaggle)

## 🌐 Giới thiệu
Dự án **Chẩn đoán viêm phổi qua ảnh X-quang** là một ứng dụng học máy (Machine Learning) được phát triển bằng Python và thư viện Scikit-learn, sử dụng tập dữ liệu ảnh X-quang ngực từ Kaggle. Hệ thống áp dụng các thuật toán phân loại để dự đoán bệnh viêm phổi (Pneumonia) dựa trên ảnh X-quang, hỗ trợ trong việc chẩn đoán y tế sớm. Dự án bao gồm trực quan hóa dữ liệu, huấn luyện mô hình và đánh giá hiệu suất.

**Tập dữ liệu**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) – Bao gồm hơn 5.800 ảnh X-quang ngực được phân loại thành **NORMAL** và **PNEUMONIA**.

## 🔑 Các chức năng chính

- **📊 Trực quan hóa dữ liệu**:
  - Thống kê số lượng ảnh
  - Biểu đồ phân bố lớp
  - Hiển thị ảnh mẫu
  - Biểu đồ histogram cường độ pixel
  - Ảnh trung bình cho mỗi lớp
  - t-SNE/PCA để phân bố đặc trưng
- **🤖 Huấn luyện & Đánh giá mô hình**:
  - Support Vector Machine (SVM)
  - Naive Bayes (Gaussian NB)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Sử dụng PCA để giảm chiều dữ liệu và StandardScaler để chuẩn hóa
  - Đánh giá bằng độ chính xác (accuracy), báo cáo phân loại (classification report), và ma trận nhầm lẫn (confusion matrix)
- **🔮 Dự đoán**:
  - Dự đoán lớp (NORMAL/PNEUMONIA) cho ảnh X-quang mới
- **📈 So sánh hiệu suất**:
  - So sánh hiệu suất các mô hình trên tập kiểm tra để chọn mô hình tốt nhất

**⚠️ Lưu ý**: Tập dữ liệu cần được tải từ Kaggle và đặt vào thư mục `chest_xray/`. Các mô hình sử dụng ảnh thang độ xám (grayscale) với kích thước 64x64 để giảm tải tính toán.

## 🏗️ Cấu trúc dự án
```
📦 Pneumonia-Detection-ML
├── 📂 SVM.py                  # Mô hình SVM: Trực quan, huấn luyện, đánh giá & dự đoán
├── 📂 Naive_Bayes.py          # Mô hình Naive Bayes: Tương tự SVM
├── 📂 KNN.py                  # Mô hình KNN: Tương tự SVM
├── 📂 Decision_Tree.py        # Mô hình Decision Tree: Tương tự SVM (bao gồm vẽ cây quyết định)
└── README.md                  # Tài liệu hướng dẫn
```

## 🛠️ Công nghệ sử dụng
- **Python 3.9+**
- **Scikit-learn**: Các thuật toán học máy (SVM, Naive Bayes, KNN, Decision Tree, PCA, t-SNE)
- **OpenCV (cv2)**: Xử lý ảnh
- **NumPy & Pandas**: Xử lý dữ liệu
- **Matplotlib & Seaborn**: Trực quan hóa dữ liệu
- **Kaggle Dataset**: Dữ liệu huấn luyện

## 🚀 Hướng dẫn cài đặt & chạy

### 1️⃣ Tải tập dữ liệu
- Truy cập [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) và tải file ZIP.
- Giải nén vào thư mục dự án với cấu trúc: `chest_xray/chest_xray` (bao gồm `train/NORMAL`, `train/PNEUMONIA`, `test/NORMAL`, `test/PNEUMONIA`, `val/`).

### 2️⃣ Cài đặt thư viện
Mở terminal và chạy:
```bash
pip install opencv-python numpy matplotlib seaborn scikit-learn
```

### 3️⃣ Chạy mã nguồn
Chạy từng file Python để huấn luyện và đánh giá mô hình:
```bash
python SVM.py
python Naive_Bayes.py
python KNN.py
python Decision_Tree.py
```
Mỗi file sẽ hiển thị biểu đồ trực quan, độ chính xác và ví dụ dự đoán.

**💡 Mẹo**: Chạy trên Google Colab hoặc Jupyter Notebook để dễ xem biểu đồ. Nếu tập dữ liệu lớn, giảm `img_size` hoặc số mẫu để tăng tốc xử lý.

## 📷 Hình ảnh minh họa (Gợi ý)
Thêm các ảnh chụp màn hình như:
- Biểu đồ phân bố lớp (tập huấn luyện)
- Ma trận nhầm lẫn (Confusion Matrix) của SVM
- Trực quan hóa t-SNE
- Kết quả dự đoán trên ảnh kiểm tra
- So sánh độ chính xác giữa các mô hình

## 📚 Tài liệu tham khảo
- Tập dữ liệu: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Tài liệu Scikit-learn: [scikit-learn.org](https://scikit-learn.org/stable/)
- Hướng dẫn OpenCV: [docs.opencv.org](https://docs.opencv.org/)
- Hướng dẫn Matplotlib: [matplotlib.org](https://matplotlib.org/stable/users/index.html)
- PCA & t-SNE: [scikit-learn.org](https://scikit-learn.org/stable/modules/decomposition.html)
- Stack Overflow: [stackoverflow.com](https://stackoverflow.com/)

## 👤 Tác giả
**Trần Quang Lâm**  
- Khoa: Công nghệ Thông tin, Đại học Đại Nam  
- Mã sinh viên: 1771020408  
- Lớp: CNTT17-11  

© 2025 Đại học Đại Nam – Môn: Nhập môn học máy