# PARKINSON-S-DISEASE-DETECTION
USING PYTHON(1D CNN )
Parkinson's Disease Prediction Using SIFT and 1D CNN

Overview
This repository contains the implementation of a Parkinson's Disease (PD) prediction system. The system uses SIFT (Scale-Invariant Feature Transform) for feature extraction and a 1D Convolutional Neural Network (1D CNN) for classification. The codebase is available in Parkinsons_Disease.py (or Parkinsons_Disease.ipynb), with the dataset in data_spiral.zip and the pre-trained model stored in the .h5 format.

Workflow
1. Image Preprocessing
Applied resizing, grayscale conversion, Gaussian blur, and normalization.
Improved feature extraction and model performance by reducing noise and normalizing intensity.
2. Feature Extraction
Tested multiple methods:
HOG (Histogram of Oriented Gradients)
SIFT (Scale-Invariant Feature Transform)
ORB (Oriented FAST and Rotated BRIEF)
Canny Edge Detection
Chose SIFT due to its superior accuracy in detecting PD-specific patterns.
3. Model Development and Optimization
Developed a 1D CNN for classification.
Performed hyperparameter tuning (e.g., learning rate, filter size, kernel size, dropout rate) to enhance accuracy and reduce overfitting.
4. Model Evaluation
Achieved 96.67% accuracy.
Evaluated using:
Accuracy
Precision
Recall
F1-Score
Cohen's Kappa
Compared the SIFT-based 1D CNN with alternatives like logistic regression and random forest ,2D CNN models.

Key Files
Parkinsons_Disease.py/Parkinsons_Disease.ipynb: Contains the Python code for preprocessing, feature extraction, model training, and evaluation.
data_spiral.zip: Dataset of spiral drawings categorized into Parkinson's and Healthy classes.
.h5: Pre-trained model file for quick deployment.
Future Scopes
Larger Dataset
Expanding the dataset to improve model generalization and robustness for diverse populations.
Stage Prediction
Enhance the model to predict different stages of Parkinson’s disease for monitoring disease progression.
Mobile Application
Develop a user-friendly mobile app for remote diagnosis and monitoring, enabling continuous patient tracking.
Advanced Feature Extraction
Investigate techniques like 3D CNNs or attention mechanisms to improve accuracy and performance.
Data Augmentation Strategies
Explore effective augmentation techniques to mitigate dataset limitations and improve training outcomes.

How to Run
Install Dependencies:
Ensure required libraries are installed 
Dataset Preparation:
Unzip data_spiral.zip into the project directory.
Run the Code:
For Python script:
python Parkinsons_Disease.py
For Jupyter Notebook: Open and execute Parkinsons_Disease.ipynb.
Model Evaluation:
The results, including metrics and visualizations, will be displayed during execution.
Results
Accuracy: 96.67%
Feature Extraction Method: SIFT
Model: 1D CNN

This project demonstrates the potential of combining advanced feature extraction with neural network architectures for accurate Parkinson's Disease detection.

Contributing
Contributions are welcome to improve this project. For major changes, please open an issue first to discuss proposed updates.

