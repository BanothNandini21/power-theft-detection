import tkinter as tk
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained models
dt_model = joblib.load('clf_dt.pkl')
knn_model = joblib.load('clf_knn.pkl')
svm_model = joblib.load('clf_svm.pkl')
rf_model = joblib.load('clf_rf.pkl')
lr_model = joblib.load('clf_lg.pkl')

# Create the GUI window
root = tk.Tk()

# Set the window title
root.title("Power Theft Detection")

# Set the window size
root.geometry("600x600")

# Create the input labels
power_label = tk.Label(root, text='Power (kWh):')
power_label.pack()
power_entry = tk.Entry(root)
power_entry.pack()
current_label = tk.Label(root, text='Current (Amp):')
current_label.pack()
current_entry = tk.Entry(root)
current_entry.pack()
voltage_label = tk.Label(root, text='Voltage (Volt):')
voltage_label.pack()
voltage_entry = tk.Entry(root)
voltage_entry.pack()


# Create the prediction button
def predict():
    # Get the input values
    power = float(power_entry.get())
    current = float(current_entry.get())
    voltage = float(voltage_entry.get())
    
    # Scale the input values
    scaler = StandardScaler()
    input_data = np.array([[power, current, voltage]])
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Debugging: Print the scaled input data
    print("Scaled Input Data:", input_data_scaled)
    
    # Make the predictions using the trained models
    dt_prediction = dt_model.predict(input_data_scaled)[0]
    knn_prediction = knn_model.predict(input_data_scaled)[0]
    svm_prediction = svm_model.predict(input_data_scaled)[0]
    rf_prediction = rf_model.predict(input_data_scaled)[0]
    lr_prediction = lr_model.predict(input_data_scaled)[0]
    
    # Debugging: Print the prediction values
    print("Decision Tree Prediction:", dt_prediction)
    print("KNN Prediction:", knn_prediction)
    print("SVM Prediction:", svm_prediction)
    print("Random Forest Prediction:", rf_prediction)
    print("Logistic Regression Prediction:", lr_prediction)
    
    # Update the prediction labels
    dt_label.config(text='Decision Tree Prediction: ' + str(dt_prediction))
    knn_label.config(text='KNN Prediction: ' + str(knn_prediction))
    svm_label.config(text='SVM Prediction: ' + str(svm_prediction))
    rf_label.config(text='Random Forest Prediction: ' + str(rf_prediction))
    lr_label.config(text='Logistic Regression Prediction: ' + str(lr_prediction))

prediction_button = tk.Button(root, text='Predict', command=predict, fg='green')
prediction_button.pack()

# Create the prediction labels
dt_label = tk.Label(root, text='Decision Tree Prediction: ')
dt_label.pack()
knn_label = tk.Label(root, text='KNN Prediction: ')
knn_label.pack()
svm_label = tk.Label(root, text='SVM Prediction: ')
svm_label.pack()
rf_label = tk.Label(root, text='Random Forest Prediction: ')
rf_label.pack()
lr_label = tk.Label(root, text='Logistic Regression Prediction: ')
lr_label.pack()

# Run the GUI window
root.mainloop()
