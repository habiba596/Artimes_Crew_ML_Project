# Student Academic Outcome Predictor

## Overview

This project predicts a student's academic outcome, Dropout, Enrolled, or Graduate, based on demographic, academic, and socioeconomic data. It was built for the EduGuard AI, Smart City AI Challenge 2026 competition on Kaggle. The final model achieved a public leaderboard score of 0.72, the second highest score recorded in the competition.

## Dataset

The dataset contains 31 features per student, covering four groups:

- Personal and demographic information: Marital status, Gender, Age at enrollment, Displaced status.
- Application and academic background: Application mode, Application order, Course, Daytime or evening attendance, Previous qualification, Admission grade.
- Family background: Mother's qualification, Father's qualification, Mother's occupation, Father's occupation.
- Academic performance and financial status: Curricular units for the 1st and 2nd semesters (credited, enrolled, evaluations, approved, grade, without evaluations), Debtor status, Tuition fees up to date, Scholarship holder, Unemployment rate, GDP.

The target variable has three classes: Dropout, Enrolled, and Graduate.

## Preprocessing

The following steps were applied to both the training and test sets:

1. Missing values were filled: categorical columns with the mode or a placeholder value, and Admission grade with the column mean.
2. Identifier and low-value columns were dropped: Application_ID, Registration_Code, International, Nacionality, Inflation rate, Previous qualification (grade), Educational special needs.
3. Text fields such as Gender and Marital status were normalized to lowercase and standardized (for example, M and F were mapped to male and female).
4. Binary fields (Displaced, Debtor, Tuition fees up to date, Scholarship holder) were standardized to yes or no, then mapped to 1 or 0.
5. Categorical columns were transformed with a One-Hot Encoder.
6. Numeric columns were scaled with a Robust Scaler, which is more resistant to outliers than standard scaling.

## Modeling

Several classical models were trained and compared as a baseline: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and AdaBoost.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.787 | 0.775 | 0.787 | 0.777 |
| K Neighbors | 0.717 | 0.705 | 0.717 | 0.702 |
| Decision Tree | 0.696 | 0.692 | 0.696 | 0.693 |
| Random Forest | 0.758 | 0.734 | 0.758 | 0.736 |
| AdaBoost | 0.754 | 0.737 | 0.754 | 0.741 |

A feed-forward neural network was then built and tuned through multiple experiments, adjusting the number of layers, the number of units per layer, the activation functions, and the dropout rates to control overfitting. The final architecture is:

- Three Dense layers of 128 units with tanh activation, each followed by a Dropout of 0.5.
- A Dense layer of 64 units with relu activation, followed by a Dropout of 0.5.
- A Dense layer of 64 units with tanh activation, followed by a Dropout of 0.5.
- A Dense layer of 32 units with relu activation, followed by a Dropout of 0.5.
- A Dense layer of 32 units with tanh activation, followed by a Dropout of 0.4.
- A Dense layer of 32 units with relu activation, followed by a Dropout of 0.3.
- An output Dense layer of 3 units with softmax activation.

The model was compiled with the Adam optimizer and sparse categorical cross-entropy loss, and trained for 100 epochs. This model produced the final Kaggle submission.

## Repository Structure

```
train_model.py       Preprocessing and training script that saves all model artifacts
app.py                Streamlit web application for interactive predictions
requirements.txt      Python dependencies
README.md             Project documentation
```

## Setup

Install the dependencies:

```
pip install -r requirements.txt
```

## Reproducing the Model

Run the training script with the path to the competition's train.csv file:

```
python train_model.py --train-path train.csv --output-dir artifacts --epochs 100
```

This creates an `artifacts` folder containing:

- model.keras, the trained neural network.
- encoder.pkl, the fitted One-Hot Encoder.
- scaler.pkl, the fitted Robust Scaler.
- feature_columns.pkl, the exact column order expected by the model.
- category_options.pkl, the valid category values for each categorical feature, used to build the dropdowns in the web app.
- cat_columns.pkl, numeric_columns.pkl, bool_columns.pkl, the feature groupings used by the app.

## Running the Web App

Once the `artifacts` folder has been generated, start the app with:

```
streamlit run app.py
```

The app presents a form with a dropdown for every categorical feature (built dynamically from the values seen during training) and a numeric input for every numeric feature. Submitting the form returns the predicted outcome along with the probability of each class.

## Notes and Limitations

- The numeric input fields do not enforce ranges based on the original data distribution. Reasonable minimum values are set where applicable, but exact bounds should be added once the real value ranges are confirmed.
- The web app depends entirely on the artifacts produced by `train_model.py`. It does not train or fit any preprocessing step on its own.
- The dropdown options for categorical features are read directly from the values observed during training, so the app remains correct even if category labels or codes change in a future version of the dataset.
