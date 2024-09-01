# Loan_Approval_Prediction_Using_ML

This project aims to predict loan approval status using machine learning algorithms. The model is trained on a dataset containing various factors that influence loan approval decisions, such as applicant income, loan amount, credit history, and more.

## Table of Contents
- #introduction
- #dataset
- #installation
- #usage
- #models-used
- #results
  
## Introduction
Loan approval is a critical process in the financial industry. This project applies machine learning to predict whether a loan will be approved or not, based on historical data. The prediction model can assist banks and financial institutions in making data-driven decisions.

## Dataset
The dataset used for this project is publicly available and contains information on loan applicants, including:
- Gender
- Married Status
- Dependency
- Education
- Self Employed
- Applicant income
- Co-applicant income
- Loan amount
- Credit history
- Property area
- Other relevant details  
  
## Installation
To run this project, you need to have Python and the following libraries installed:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install these libraries using:
```bash 
pip install pandas numpy scikit-learn matplotlib seaborn

## Usage

To use this project, follow these steps:

### 1. Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/Ravi7264/Loan_Approval_Prediction_Using_ML.git

cd Loan_Approval_Prediction_Using_ML
pip install pandas numpy scikit-learn matplotlib seaborn

To make loan approval predictions, run the main script:
jupyter notebook Loan_Approval_Prediction.ipynb

##Model Used
The following machine learning models were explored in this project:
- Random Forest
- Gaussian Naive Bayes 
- Decision Tree
- KNeighborsClassifier

These above mmodels are used for the accuracy prediction.


## Results

The results of this project demonstrate the effectiveness of various machine learning models in predicting loan approval status. Below is a summary of the model performances:

### Model Performance

| Model                        | Accuracy | 
|------------------------------|----------|
| Random Forest                | 76%      |
| Gaussian Naive Bayes         | 82%      | 
| Decision Tree                | 72%      | 
| SKNeighborsClassifier        | 79%      | 

### Confusion Matrix

Below is an example of a confusion matrix for the best-performing model (Random Forest):

|                 | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | 82                 | 18                 |
| Actual Negative | 20                 | 80                 |

### Key Insights
- **Gaussian Naive Bayes** achieved the highest accuracy (82%), making it the most reliable model for this dataset.
- **SKNeighborsClassifier** also performed well with an accuracy of 79%, offering a simpler model with good interpretability.
- **Random Forest** and **Decision Tree** showed competitive performance but were slightly less accurate than Gaussian Naive Bayes.

### ROC Curve
The Receiver Operating Characteristic (ROC) curve shows the trade-off between sensitivity and specificity for the models. The Area Under the Curve (AUC) for the Gaussian Naive Bayes was 0.85, indicating good model performance.


### Conclusion
The Gaussian Naive Bayes is recommended for predicting loan approval status due to its high accuracy and balanced performance metrics. The model can be further improved by tuning hyperparameters or exploring additional features.




