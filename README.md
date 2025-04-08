# Data Science Final Project: E-commerce Product Image Classification

## 1. Project Overview

This project focuses on developing a machine learning model to automatically classify e-commerce product images into five distinct categories: **Kurti, Men's Shirt, Saree, Women's Top & Tunic, and Women's T-Shirt**. The primary goal is to minimize the manual effort required for product cataloging in e-commerce platforms, thereby enhancing efficiency and accuracy. This addresses common issues like mismatches between product images and their descriptions.

## 2. Team Members

*   Saksham Singh (2022434)
*   Swarnima Prasad (2022525)
*   Ritika Thakur (2022468)
*   Sidhartha Garg (2022499)

*(CSE558 Data Science Group Project)*

## 3. Dataset

*   **Source:** E-commerce product dataset containing product information and high-quality images.
*   **Size:** Approximately 100,000 image files split into training and testing sets.
*   **Attributes:**
    *   `Product ID`: Unique identifier.
    *   `Product Image`: URLs or local paths (`productid.jpg`).
    *   `Category`: Type of garment (target variable).
    *   `Attribute Keys`: Names of attributes specified for the product (e.g., 'color', 'sleeve_length').
    *   `Attribute Values`: Values corresponding to the attribute keys.
*   **Metadata:** Provided via CSV files (train/test attributes) and a Parquet file describing the attributes.

## 4. Methodology

The project followed a standard data science workflow:

### 4.1. Exploratory Data Analysis (EDA)

Performed EDA to understand data patterns, distributions, relationships, and anomalies. This included:
*   **Attribute Analysis:** Examining class distributions, common attributes per category, missing values, and attribute value frequencies using tools like `matplotlib` and `seaborn`.
*   **Image Analysis:** Analyzing image dimensions (width, height, aspect ratio), color channel distributions, and identifying potential issues like blurriness.

### 4.2. Preprocessing

Several steps were taken to prepare the data for modeling:
*   **Image Resizing:** Standardized images to 360x360 pixels (scaling/padding) for consistent model input.
*   **Feature Extraction:**
    *   **Bounding Box & Silhouette:** Extracted object contours to isolate products from the background.
    *   **Color Histograms:** Computed normalized RGB color distributions (20 bins/channel).
    *   **(For RF/MLP) Texture/Shape Features:** HOG, Gabor, and Edge features were extracted (totaling 14,176 features for a 128x128 image).
*   **Dimensionality Reduction (PCA):** Applied Incremental PCA to the HOG/Gabor/Edge features, reducing them to 100 principal components (~72% variance retained) for use with Random Forest and MLP models.

### 4.3. Hypothesis Testing

Statistical tests were conducted to validate assumptions and gain insights:
*   **Hypothesis 1 (Bounding Box Variance/Means):** Used Levene's test to check if bounding box consistency (variance) and average position/size (mean) differed significantly across categories. *Result: Rejected null hypothesis - significant differences exist.*
*   **Hypothesis 2 (Color Histogram Consistency - Attributes):** Used Kruskal-Wallis H-test to check if RGB color channel distributions differed significantly *from each other*. *Result: Rejected null hypothesis - R, G, B distributions are distinct.*
*   **Hypothesis 3 (Missing Data Completeness):** Used Chi-square test to determine if missing attribute values were equally distributed across categories. *Result: Rejected null hypothesis - missingness varies by category.*
*   **Hypothesis 4 (Color Histogram Consistency - Categories):** Used Kruskal-Wallis H-test to check if RGB color distributions differed significantly *across product categories*. *Result: Rejected null hypothesis - color distributions vary significantly between categories and can be used as features.*

### 4.4. ML Modelling

Three different models were trained and evaluated for the classification task:
*   **Random Forest (RF):** Ensemble method using the 100 PCA features derived from HOG/Gabor/Edge.
*   **Multi-Layer Perceptron (MLP):** Neural network using the 100 PCA features.
*   **Convolutional Neural Network (CNN):** Deep learning model specialized for image data, trained directly on the resized images.

Models were initially tested on undersampled data and then trained/evaluated on the full dataset.

## 5. Results

Model performance on the **full dataset**:

| Model                             | Feature Set        | Accuracy | Notes                                      |
| :-------------------------------- | :----------------- | :------- | :----------------------------------------- |
| Random Forest (RF)                | 100 PCA Components | 84%      | Good baseline, handles structured data well. |
| Multi-Layer Perceptron (MLP)      | 100 PCA Components | 85%      | Slightly better than RF on PCA features.   |
| **Convolutional Neural Network (CNN)** | **Raw Images**     | **88%**  | **Best performing model.**                   |

CNN achieved the highest accuracy, demonstrating its effectiveness in capturing spatial and hierarchical features directly from images for this task.

## 6. Conclusion

The project successfully demonstrated the feasibility of automating e-commerce product image classification using machine learning. Key achievements include:
*   Thorough dataset analysis and hypothesis testing to understand data characteristics.
*   Implementation of relevant preprocessing steps including feature engineering and dimensionality reduction.
*   Successful training and evaluation of Random Forest, MLP, and CNN models.
*   Achieving a **peak accuracy of 88%** using a CNN model for classifying images into five categories.

## 7. Future Work

*   **Multi-Label Classification:** Extend the model to predict multiple attributes (e.g., color, pattern, sleeve length) simultaneously.
*   **Real-Time Search:** Enable real-time product search across e-commerce platforms based on image features.
*   **Scalability:** Adopt distributed training techniques for faster scaling on larger datasets.
*   **Deployment:** Deploy the model for automated, real-time product cataloging.

## 8. Dependencies & References

*   Python
*   Libraries:
    *   `scikit-learn` (PCA, RF, MLP, Metrics) - [Docs](https://scikit-learn.org/1.5/index.html)
    *   `PyTorch` or `TensorFlow/Keras` (CNN, MLP) - [PyTorch Docs](https://pytorch.org/docs/stable/index.html) | [TensorFlow Guide](https://www.tensorflow.org/guide/keras)
    *   `SciPy` (Stats, Hypothesis Testing) - [Docs](https://docs.scipy.org/doc/scipy/reference/stats.html)
    *   `Pandas` (Data handling)
    *   `NumPy` (Numerical operations)
    *   `Matplotlib`, `Seaborn` (Visualizations)
    *   `OpenCV` (Image processing)
*   Statistical Reference: [MedCalc Tables](https://www.medcalc.org/manual/statistical-tables.php)
