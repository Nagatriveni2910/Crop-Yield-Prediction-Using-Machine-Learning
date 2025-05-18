from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle

# Initialize main GUI window
main = tkinter.Tk()
main.title("Crop Yield Prediction using Machine Learning Algorithm")
main.geometry("1600x1100")

# Global variables
global filename
global X_train, X_test, y_train, y_test
global X, Y
global dataset
global le
global model

def upload():
    """Function to upload dataset and display its contents."""
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.insert(END, 'Crop dataset loaded\n')
    dataset = pd.read_csv(filename)
    dataset.columns = dataset.columns.str.strip()
    dataset.fillna(0, inplace=True)
    if 'Production' in dataset.columns:
        dataset['Production'] = dataset['Production'].astype(np.int64)
    text.insert(END, "Dataset Preview:\n" + str(dataset.head(10)) + "\n")

def processDataset():
    """Function to preprocess dataset: encoding categorical variables and normalizing features."""
    global le
    global dataset
    global X_train, X_test, y_train, y_test
    global X, Y
    text.delete('1.0', END)
    if dataset is None or dataset.empty:
        text.insert(END, "Dataset not loaded. Please upload a dataset first.\n")
        return
    
    le = LabelEncoder()
    required_columns = ['State_Name', 'District_Name', 'Season', 'Crop']
    for col in required_columns:
        if col in dataset.columns:
            dataset[col] = le.fit_transform(dataset[col])
    
    datasets = dataset.values
    X = datasets[:, :-1]
    Y = datasets[:, -1]
    Y = Y.astype('uint8')
    X = normalize(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    text.insert(END, f"Total records: {len(X)}\nTraining set: {X_train.shape[0]}\nTesting set: {X_test.shape[0]}\n")

def trainModel():
    """Function to train Random Forest model and calculate error rate."""
    global model
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    model = RandomForestRegressor(n_estimators=100, max_depth=40, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    
    with open('rf_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    mse = mean_squared_error(y_test, predict)
    rmse = np.sqrt(mse)
    text.insert(END, f"Model trained successfully. RMSE Error: {rmse:.4f}\n")

def cropYieldPredict():
    """Function to load test data, predict crop yield, and display original categorical values."""
    global model
    global le
    text.delete('1.0', END)
    testname = filedialog.askopenfilename(initialdir="Dataset")
    test = pd.read_csv(testname)
    test.columns = test.columns.str.strip()
    test.fillna(0, inplace=True)

    # Store original categorical values
    original_values = test[['State_Name', 'District_Name', 'Season', 'Crop']].copy()

    required_columns = ['State_Name', 'District_Name', 'Season', 'Crop']
    for col in required_columns:
        if col in test.columns:
            test[col] = test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            test[col] = le.transform(test[col])

    test = test.values
    test = normalize(test)
    
    predict = model.predict(test)
    
    text.insert(END, "Predicted Crop Yield:\n")
    for i in range(len(predict)):
        production = predict[i] * 100
        crop_yield = production / 10000
        text.insert(END, f"Test Record {i+1}: State: {original_values.iloc[i]['State_Name']}, "
                         f"District: {original_values.iloc[i]['District_Name']}, "
                         f"Season: {original_values.iloc[i]['Season']}, "
                         f"Crop: {original_values.iloc[i]['Crop']}, "
                         f"Production: {production:.2f} KGs, Yield: {crop_yield:.4f} KGs/acre\n")


def close():
    """Function to close the application."""
    main.destroy()

# GUI Components
font = ('times', 16, 'bold')
title = Label(main, text='Crop Yield Prediction using Machine Learning Algorithm')
title.config(bg='dark goldenrod', fg='white', font=font, height=2, width=120)
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Crop Dataset", command=upload)
upload.place(x=700, y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white', font=font1)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=700, y=150)
processButton.config(font=font1)

mlButton = Button(main, text="Train Machine Learning Algorithm", command=trainModel)
mlButton.place(x=700, y=200)
mlButton.config(font=font1)

predictButton = Button(main, text="Upload Test Data & Predict Yield", command=cropYieldPredict)
predictButton.place(x=700, y=250)
predictButton.config(font=font1)

closeButton = Button(main, text="Close", command=close)
closeButton.place(x=700, y=300)
closeButton.config(font=font1)

text = Text(main, height=30, width=80)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=100)
text.config(font=('times', 12, 'bold'))

main.config(bg='turquoise')
main.mainloop()
