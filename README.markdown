# Pneumonia Detection using Machine Learning

![Made by Trần Quang Lâm](https://img.shields.io/badge/Made%20by%20Trần%20Quang%20Lâm-blue?style=for-the-badge)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-orange?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?style=for-the-badge&logo=scikit-learn)
![Kaggle Dataset](https://img.shields.io/badge/Kaggle%20Dataset-red?style=for-the-badge&logo=kaggle)

## 🌐 Introduction
The **Pneumonia Detection using Machine Learning** project is a machine learning application developed in Python using Scikit-learn, leveraging a chest X-ray dataset from Kaggle. The system employs classification algorithms to predict pneumonia from chest X-ray images, aiding early medical diagnosis. The project includes data visualization, model training, and performance evaluation.

**Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) – Contains over 5,800 chest X-ray images classified as **NORMAL** or **PNEUMONIA**.

## 🔑 Key Features

- **📊 Data Visualization**:
  - Statistics on image counts
  - Class distribution charts
  - Sample image display
  - Pixel intensity histograms
  - Average image per class
  - t-SNE/PCA for feature distribution
- **🤖 Model Training & Evaluation**:
  - Support Vector Machine (SVM)
  - Naive Bayes (Gaussian NB)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Uses PCA for dimensionality reduction and StandardScaler for normalization
  - Evaluated with accuracy, classification report, and confusion matrix
- **🔮 Prediction**:
  - Predicts class (NORMAL/PNEUMONIA) for new X-ray images
- **📈 Performance Comparison**:
  - Compares model performance on the test set to identify the best model

**⚠️ Note**: The dataset must be downloaded from Kaggle and placed in the `chest_xray/` directory. Models use grayscale images resized to 64x64 to reduce computational load.

## 🏗️ Project Structure
```
📦 Pneumonia-Detection-ML
├── 📂 SVM.py                  # SVM model: Visualization, training, evaluation & prediction
├── 📂 Naive_Bayes.py          # Naive Bayes model: Similar to SVM
├── 📂 KNN.py                  # KNN model: Similar to SVM
├── 📂 Decision_Tree.py        # Decision Tree model: Similar to SVM (includes tree visualization)
└── README.md                  # Project documentation
```

## 🛠️ Technologies Used
- **Python 3.9+**
- **Scikit-learn**: ML algorithms (SVM, Naive Bayes, KNN, Decision Tree, PCA, t-SNE)
- **OpenCV (cv2)**: Image processing
- **NumPy & Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Kaggle Dataset**: Training data

## 🚀 Installation & Setup

### 1️⃣ Download Dataset
- Visit the [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and download the ZIP file.
- Extract it into the project directory with the structure: `chest_xray/chest_xray` (including `train/NORMAL`, `train/PNEUMONIA`, `test/NORMAL`, `test/PNEUMONIA`, `val/`).

### 2️⃣ Install Dependencies
Open a terminal and run:
```bash
pip install opencv-python numpy matplotlib seaborn scikit-learn
```

### 3️⃣ Run the Code
Execute each Python file to train and evaluate the models:
```bash
python SVM.py
python Naive_Bayes.py
python KNN.py
python Decision_Tree.py
```
Each script will display visualizations, accuracy metrics, and prediction examples.

**💡 Tip**: Use Google Colab or Jupyter Notebook for easier visualization. For large datasets, reduce `img_size` or sample size to speed up processing.

## 📷 Visualizations (Suggestions)
Include screenshots such as:
- Class distribution chart (train set)
- Confusion Matrix for SVM
- t-SNE visualization
- Prediction results for test images
- Accuracy comparison across models

## 📚 References
- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Scikit-learn Docs: [scikit-learn.org](https://scikit-learn.org/stable/)
- OpenCV Tutorial: [docs.opencv.org](https://docs.opencv.org/)
- Matplotlib Guide: [matplotlib.org](https://matplotlib.org/stable/users/index.html)
- PCA & t-SNE: [scikit-learn.org](https://scikit-learn.org/stable/modules/decomposition.html)
- Stack Overflow: [stackoverflow.com](https://stackoverflow.com/)

## 👤 Author
**Trần Quang Lâm**  
- Faculty: Information Technology, Đại Nam University  
- Student ID: 1771020408  
- Class: CNTT17-11  

© 2025 Đại Nam University – Course: Introduction to Machine Learning