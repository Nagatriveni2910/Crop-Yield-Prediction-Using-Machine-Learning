**#Project Name**

🌾 Crop Yield Prediction using Machine Learning

This project is a GUI-based Machine Learning application built with Tkinter to predict crop yield based on agricultural data. It uses Random Forest Regression to train models on agricultural datasets and predict production/yield. The application is designed to be simple, user-friendly, and useful for farmers and researchers to make data-driven decisions.

**##🚀 Features**

* Upload agricultural dataset and preview data.

* Preprocess dataset (handle missing values, label encoding, normalization).

* Train a Random Forest Regression model with high accuracy.

* Predict crop production and yield by uploading a test dataset.

* Display original categorical data (State, District, Season, Crop) along with predictions.

* User-friendly Tkinter GUI for non-technical users.


**##🛠️ Tech Stack**

* **Programming Language:**Python

* **GUI Framework:** Tkinter

* **Machine Learning Libraries:**

      * scikit-learn (RandomForestRegressor, DecisionTreeRegressor, LabelEncoder, Normalization)

      * pandas, numpy, matplotlib

* **Other Tools:** pickle (for saving trained model)

* **Dataset:** Custom agricultural crop production dataset (CSV files)

**##📂 Project Structure**

├── CropYield.py         # Main application with Tkinter GUI
├── test_data.py         # Script to test ML model with test dataset
├── Dataset/             # Folder containing training & test CSV files
├── rf_model.pkl         # Trained Random Forest model (generated after training)
└── README.md            # Project documentation

**##📊 Dataset Format**

Your dataset should include the following columns:

| State\_Name    | District\_Name | Season | Crop      | Area | Production |
| -------------- | -------------- | ------ | --------- | ---- | ---------- |
| Andhra Pradesh | Guntur         | Kharif | Rice      | 5000 | 12000      |
| Karnataka      | Mysore         | Rabi   | Wheat     | 3000 | 8000       |
| Maharashtra    | Pune           | Summer | Sugarcane | 2000 | 9000       |
| Punjab         | Amritsar       | Kharif | Maize     | 2500 | 7000       |
| Tamil Nadu     | Coimbatore     | Winter | Cotton    | 1500 | 5000       |

**###💡 Tip:**

* Production should be numeric.

* Missing values should be replaced with 0 or appropriate defaults.

* Save your dataset as Dataset/Dataset.csv.

* Create a smaller test file like Dataset/test.csv for predictions.

**##▶️ How to Run**

**###1️⃣ Clone the Repository**

```bash

git clone https://github.com/Nagatriveni2910/Crop-Yield-Prediction-Using-Machine-Learning.git

**###2️⃣ Navigate to the Project Directory**

cd crop-yield-prediction

**###Install Dependencies**

pip install pandas numpy scikit-learn matplotlib

