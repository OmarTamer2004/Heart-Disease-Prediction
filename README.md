# Heart Disease Prediction

This project provides a machine learning pipeline for predicting the likelihood of heart disease using a dataset of clinical features. It includes data preprocessing, model training, evaluation, and prediction steps. The main code is implemented inside the provided Jupyter Notebook `Heart_Disease_Prediction.ipynb`.

---

## 📌 Project Overview

The goal of this project is to build a predictive model using logistic regression (or alternative ML algorithms) to classify whether a patient may have heart disease based on medical attributes.

The notebook includes:

* Data loading and exploration
* Data preprocessing and feature engineering
* Model training using scikit-learn
* Performance evaluation
* Making predictions on new input data

---

## 📁 Files in This Repository

* **Heart_Disease_Prediction.ipynb** — The main Jupyter Notebook with the complete workflow.
* **README.md** — Documentation for the project.

(You can add more files later such as saved models, datasets, or scripts.)

---

## 🧪 Requirements

To run the notebook, install the following Python libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Optionally:

```bash
pip install jupyter
```

---

## 🚀 How to Run the Project

1. Clone this repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
```

2. Navigate to the project folder:

```bash
cd your-repo-name
```

3. Open the Jupyter Notebook:

```bash
jupyter notebook Heart_Disease_Prediction.ipynb
```

4. Run each cell sequentially to reproduce the training and predictions.

---

## 🧠 Model Details

The notebook uses **Logistic Regression**, a common and effective algorithm for binary classification.

If you encounter this error:

```
NotFittedError: This LogisticRegression instance is not fitted yet.
```

It means you need to call:

```python
model.fit(X_train, y_train)
```

before making predictions.

---

## 📊 Dataset

If your dataset is not included in the repository, place it in the working directory or modify the notebook path accordingly.

Make sure the dataset contains the expected columns such as:

* Age
* Sex
* ChestPainType
* RestingBP
* Cholesterol
* FastingBS
* RestingECG
* MaxHR
* ExerciseAngina
* Oldpeak
* ST_Slope

---

## 🤖 Making Predictions

Once the model is trained, you can run:

```python
pred = model.predict(input_data_reshaped)
print(pred)
```

Make sure the model is fitted and the input data is reshaped properly.

---

## 📌 Future Improvements

* Add additional ML models (Random Forest, SVM, XGBoost)
* Hyperparameter tuning
* Improve dataset preprocessing
* Deploy model as API or web app

---

## 📝 License

This project is open-source. You may modify and use it freely.

---

## 📬 Contact

If you have questions or suggestions, feel free to open an issue or submit a pull request!
