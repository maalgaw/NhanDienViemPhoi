# 1. Import thư viện
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE

# 2. Đường dẫn dataset
data_dir = "chest_xray/chest_xray"   # chỉnh lại theo thư mục của bạn
splits = ["train", "val", "test"]
categories = ["NORMAL", "PNEUMONIA"]

# ======================================================
# PHẦN 1: TRỰC QUAN HÓA DỮ LIỆU
# ======================================================

# B1. Thống kê số lượng ảnh
print("Thống kê số lượng ảnh:")
for split in splits:
    print(f"--- {split.upper()} ---")
    for category in categories:
        path = os.path.join(data_dir, split, category)
        print(category, ":", len(os.listdir(path)))
    print()

# B2. Biểu đồ phân bố lớp (train set)
train_counts = {cat: len(os.listdir(os.path.join(data_dir, "train", cat))) for cat in categories}
sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()))
plt.title("Số lượng ảnh trong tập Train")
plt.ylabel("Số ảnh")
plt.show()

# B3. Hiển thị ảnh mẫu
plt.figure(figsize=(8, 4))
for i, category in enumerate(categories):
    path = os.path.join(data_dir, "train", category)
    img_name = random.choice(os.listdir(path))
    img_path = os.path.join(path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.subplot(1, 2, i+1)
    plt.imshow(img, cmap="gray")
    plt.title(category)
    plt.axis("off")
plt.suptitle("Ảnh mẫu trong tập Train", fontsize=14)
plt.show()

# B4. Histogram pixel intensity
def get_pixels(path, n=100):
    files = os.listdir(path)
    sample = random.sample(files, min(n, len(files)))
    pix = []
    for f in sample:
        img = cv2.imread(os.path.join(path,f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            pix.extend(img.ravel())
    return np.array(pix)

pix_normal = get_pixels(os.path.join(data_dir,"train","NORMAL"))
pix_pneumonia = get_pixels(os.path.join(data_dir,"train","PNEUMONIA"))

plt.hist(pix_normal, bins=50, alpha=0.5, label="Normal")
plt.hist(pix_pneumonia, bins=50, alpha=0.5, label="Pneumonia")
plt.legend()
plt.title("Histogram cường độ pixel (100 ảnh mẫu)")
plt.xlabel("Mức xám (0-255)")
plt.ylabel("Tần suất")
plt.show()

# B5. Ảnh trung bình mỗi lớp
def mean_image(path, size=(64,64), n=200):
    files = os.listdir(path)
    sample = random.sample(files, min(n, len(files)))
    acc = np.zeros(size, dtype=np.float32)
    for f in sample:
        img = cv2.imread(os.path.join(path,f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, size)
            acc += img
    return acc / len(sample)

mean_normal = mean_image(os.path.join(data_dir,"train","NORMAL"))
mean_pneu = mean_image(os.path.join(data_dir,"train","PNEUMONIA"))

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(mean_normal, cmap="gray"); plt.title("Mean - Normal"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(mean_pneu, cmap="gray"); plt.title("Mean - Pneumonia"); plt.axis("off")
plt.suptitle("Ảnh trung bình theo lớp", fontsize=14)
plt.show()

# B6. PCA + t-SNE trực quan hóa đặc trưng
sample_data, sample_labels = [], []
for cls in categories:
    path = os.path.join(data_dir,"train",cls)
    files = os.listdir(path)
    for f in random.sample(files, min(200, len(files))):  # 200 ảnh mỗi lớp
        img = cv2.imread(os.path.join(path,f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64,64)).flatten()
            sample_data.append(img)
            sample_labels.append(cls)

X_vis = np.array(sample_data)
y_vis = np.array(sample_labels)

X_pca = PCA(n_components=50).fit_transform(X_vis)
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_pca)

plt.figure(figsize=(6,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_vis, palette={"NORMAL":"blue","PNEUMONIA":"red"}, alpha=0.7)
plt.title("Phân bố dữ liệu bằng t-SNE")
plt.show()

# ======================================================
# PHẦN 2: HUẤN LUYỆN & ĐÁNH GIÁ VỚI KNN
# ======================================================

# Tiền xử lý ảnh train
img_size = 64
data, labels = [], []
for category in categories:
    path = os.path.join(data_dir, "train", category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size, img_size))
            data.append(new_array.flatten())
            labels.append(class_num)
        except:
            pass

X = np.array(data)
y = np.array(labels)
print("Tổng số ảnh train:", len(X))

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Huấn luyện KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)

# Dự đoán
y_pred = knn.predict(X_test_pca)

# Đánh giá
print("Độ chính xác:", accuracy_score(y_test, y_pred))
print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred, target_names=categories))

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix - KNN")
plt.colorbar()
tick_marks = np.arange(len(categories))
plt.xticks(tick_marks, categories)
plt.yticks(tick_marks, categories)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
# ======================================================
# PHẦN 3: DỰ ĐOÁN TRÊN ẢNH MỚI
# ======================================================

def predict_image(img_path, model, scaler, pca, img_size=64):
    """
    Hàm dự đoán xem ảnh X-quang là Normal hay Pneumonia
    """
    # Đọc ảnh
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Không thể đọc ảnh:", img_path)
        return None

    # Tiền xử lý ảnh
    img_resized = cv2.resize(img, (img_size, img_size))
    img_flatten = img_resized.flatten().reshape(1, -1)

    # Chuẩn hóa + PCA giống như lúc train
    img_scaled = scaler.transform(img_flatten)
    img_pca = pca.transform(img_scaled)

    # Dự đoán
    pred = model.predict(img_pca)[0]
    label = categories[pred]

    # Hiển thị ảnh và kết quả
    plt.imshow(img_resized, cmap="gray")
    plt.title(f"Dự đoán: {label}")
    plt.axis("off")
    plt.show()

    return label

# Ví dụ sử dụng:
test_img_path = os.path.join(data_dir, "test", "PNEUMONIA", random.choice(os.listdir(os.path.join(data_dir, "test", "PNEUMONIA"))))
print("Ảnh test:", test_img_path)
predict_image(test_img_path, knn, scaler, pca)
