# FireDetect: A Machine Learning Approach for Fire Classification

## Project Overview
FireDetect is a machine learning-based classification project that aims to differentiate between fire and no-fire scenarios using multiple classification models. The project evaluates the performance of various machine learning algorithms before and after hyperparameter tuning to determine the best-performing model.

## Features
- **Data Preprocessing:** Standardization and feature extraction using TF-IDF (if applicable to text-based data).
- **Model Training:** Implementation of multiple classifiers including:
  - Support Vector Machine (SVC)
  - Logistic Regression
  - Decision Tree Classifier
  - Multi-Layer Perceptron (MLP)
  - Random Forest Classifier
- **Model Evaluation:** Performance comparison using accuracy, precision, recall, F1-score, and confusion matrices.
- **Hyperparameter Tuning:** Optimization using GridSearchCV and RandomizedSearchCV.
- **Visualization:** Confusion matrices and comparison plots generated using Seaborn and Matplotlib.

## Installation
Ensure you have Python installed along with the required dependencies. You can install them using:
```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

## Usage
1. Load the dataset and preprocess it.
2. Split the dataset into training and testing sets.
3. Train various machine learning models.
4. Evaluate and compare their performance.
5. Tune hyperparameters and observe improvements.
6. Visualize results.

## Performance Comparison
Before and after tuning, the models are evaluated on:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## License
This project is open-source and available under the MIT License.

## Contact
For questions or contributions, feel free to reach out or open an issue in the repository.
