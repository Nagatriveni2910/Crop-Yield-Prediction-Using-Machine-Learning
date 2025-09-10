<div align="left">

# Project Name

### 🌾 Crop Yield Prediction using Machine Learning

This project is a GUI-based Machine Learning application built with Tkinter to predict crop yield based on agricultural data. It uses Random Forest Regression to train models on agricultural datasets and predict production/yield. The application is designed to be simple, user-friendly, and useful for farmers and researchers to make data-driven decisions.

## 🚀 Features

* Upload agricultural dataset and preview data.

* Preprocess dataset (handle missing values, label encoding, normalization).

* Train a Random Forest Regression model with high accuracy.

* Predict crop production and yield by uploading a test dataset.

* Display original categorical data (State, District, Season, Crop) along with predictions.

* User-friendly Tkinter GUI for non-technical users.


## 🛠️ Tech Stack

* **Programming Language:** Python

* **GUI Framework:** Tkinter

* **Machine Learning Libraries:**

   * scikit-learn (RandomForestRegressor, DecisionTreeRegressor, LabelEncoder, Normalization)

    * pandas, numpy, matplotlib

* **Other Tools:** pickle (for saving trained model)

* **Dataset:** Custom agricultural crop production dataset (CSV files)

## 📂 Project Structure

├── CropYield.py         # Main application with Tkinter GUI

├── test_data.py         # Script to test ML model with test dataset

├── Dataset/             # Folder containing training & test CSV files

├── rf_model.pkl         # Trained Random Forest model (generated after training)

└── README.md            # Project documentation

## 📊 Dataset Format

Your dataset should include the following columns:

| State\_Name    | District\_Name | Season | Crop      | Area | Production |
| -------------- | -------------- | ------ | --------- | ---- | ---------- |
| Andhra Pradesh | Guntur         | Kharif | Rice      | 5000 | 12000      |
| Karnataka      | Mysore         | Rabi   | Wheat     | 3000 | 8000       |
| Maharashtra    | Pune           | Summer | Sugarcane | 2000 | 9000       |
| Punjab         | Amritsar       | Kharif | Maize     | 2500 | 7000       |
| Tamil Nadu     | Coimbatore     | Winter | Cotton    | 1500 | 5000       |

### 💡 Tip:

* Production should be numeric.

* Missing values should be replaced with 0 or appropriate defaults.

* Save your dataset as Dataset/Dataset.csv.

* Create a smaller test file like Dataset/test.csv for predictions.

## ▶️ How to Run

### 1️⃣ Clone the Repository

```bash

git clone https://github.com/Nagatriveni2910/Crop-Yield-Prediction-Using-Machine-Learning.git
```

### 2️⃣ Navigate to the Project Directory

```bash

cd crop-yield-prediction
```

### 3️⃣ Install Dependencies

```bash

```pip install pandas numpy scikit-learn matplotlib```
```

### 4️⃣ Run the Project

```bash
python CropYield.py
```

## 🧪 Testing

The project has been tested for:

* Dataset loading and preprocessing.

* ML model training and prediction accuracy.

* GUI functionality (upload, train, predict).

* Error handling for missing/wrong inputs.

## 🔮 Future Enhancements

* Integrate real-time weather and soil APIs for better predictions.

* Add fertilizer recommendation system.

* Create a mobile version for farmer-friendly accessibility.

* Deploy the model as a web app for wider reach.

## 🖼️ Screenshot

### Output: 

<img width="1920" height="1080" alt="Screenshot 2025-03-26 000624" src="https://github.com/user-attachments/assets/f31d0f6a-0f7a-4c45-96a4-63291b40c3ba" />

## 💡 Conclusion

This project demonstrates how machine learning can assist in agriculture by providing accurate crop yield predictions in a simple GUI. It’s a step towards precision farming, enabling farmers and stakeholders to make informed decisions based on data.

</div>
