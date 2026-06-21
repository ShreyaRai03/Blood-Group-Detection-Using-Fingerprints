#  Blood Group Detection Using Fingerprint Analysis

##  Overview
This project applies Machine Learning and Image Processing techniques to predict a person’s blood group using fingerprint patterns. The system extracts meaningful features from fingerprint images and uses classification algorithms to identify blood group categories.

This project explores the potential relationship between biometric fingerprint features and blood group classification using data-driven approaches.

---

##  Objective
To develop a machine learning-based system that analyzes fingerprint structures and predicts blood groups, demonstrating the application of biometrics in classification problems.

---

##  Technologies
- Python  
- OpenCV (Image Processing)  
- NumPy (Numerical Computation)  
- Scikit-learn (Machine Learning)  
- Machine Learning Algorithms  

---

##  Workflow

1. Load fingerprint dataset  
2. Preprocess images (grayscale conversion, noise reduction)  
3. Extract fingerprint features  
4. Train machine learning model  
5. Evaluate model performance  
6. Predict blood group from new fingerprint input  

---

##  Machine Learning Model
The project uses classification algorithms such as:
- Support Vector Machine (SVM)  
- Random Forest Classifier  

These models are trained on extracted fingerprint features to classify blood group categories.

---

##  Project Structure

```
BloodGroupDetection/
│
├── dataset/ # Fingerprint dataset
├── models/ # Trained ML models
├── main.py # Main execution file
├── requirements.txt # Dependencies
└── README.md # Project documentation
```


---

##  How to Run the Project

##### Clone the repository
```bash
git clone https://github.com/ShreyaRai03/BloodGroupDetection.git
```

##### Navigate to project directory
```bash
cd BloodGroupDetection
```

#####  Install dependencies
```bash
pip install -r requirements.txt
```

#####  Run the application
```bash
python main.py
```

## Results

The trained model is capable of classifying fingerprint patterns into corresponding blood group categories based on learned features from the dataset.

## Future Improvements
- Improve accuracy using Deep Learning (CNN-based models)
- Expand dataset for better generalization
- Deploy as a web application using Flask or Streamlit
- Enhance feature extraction using advanced image processing techniques

## Key Learnings
- Image preprocessing using OpenCV
- Feature extraction from biometric data
- Supervised machine learning classification
- Model training and evaluation pipeline

## Author

Shreya

Aspiring Software Engineer | Machine Learning Enthusiast | Python Developer

## If you like this project

Give it a ⭐ on GitHub and feel free to contribute or improve it!
