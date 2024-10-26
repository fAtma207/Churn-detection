import pandas as pd
import numpy as np
import joblib
import gradio as gr


# Load the preprocessing steps and the model
label_encoders = joblib.load('model/label_encoders.pkl')
one_hot_encoder = joblib.load('model/one_hot_encoder.pkl')
min_max_scaler = joblib.load('model/min_max_scaler.pkl')
model = joblib.load('model/logistic_regression_model.pkl')
le_target = joblib.load('model/label_encoder_target.pkl')


def preprocess_data(data):
    """
    Preprocess the input data for prediction.

    Parameters:
    data (dict): Dictionary containing input data.

    Returns:
    np.array: Processed data ready for prediction.
    """
    df = pd.DataFrame([data])

    label_encode_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "gender"]
    one_hot_encode_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                           "Contract", "PaymentMethod"]
    min_max_scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Strip leading and trailing spaces from string inputs
    for col in label_encode_cols + one_hot_encode_cols:
        df[col] = df[col].str.strip()

    # Convert non-numeric values to NaN and fill them with the mean of the column
    df[min_max_scale_cols] = df[min_max_scale_cols].replace(' ', np.nan).astype(float)
    df[min_max_scale_cols] = df[min_max_scale_cols].fillna(df[min_max_scale_cols].mean())

    # Label encode specified columns
    for col in label_encode_cols:
        le = label_encoders[col]
        df[col] = le.transform(df[col])


    # One-hot encode specified columns
    one_hot_encoded = one_hot_encoder.transform(df[one_hot_encode_cols])

    # Min-max scale specified columns
    scaled_numerical = min_max_scaler.transform(df[min_max_scale_cols])

    # Combine processed columns into one DataFrame
    X_processed = np.hstack((df[label_encode_cols].values, scaled_numerical, one_hot_encoded))
    print(df.head())
    return X_processed


def predict(gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service,
            online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies,
            contract, paperless_billing, payment_method, monthly_charges, total_charges):
    """
    Predict the churn status of a customer.

    Parameters:
    Various input features as separate parameters.

    Returns:
    str: Prediction result ("Churn" or "No Churn").
    """

    data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        X_new = preprocess_data(data)
        prediction = model.predict(X_new)
        prediction = le_target.inverse_transform(prediction)
        return "Churn" if prediction[0] == 'Yes' else "No Churn"
    except Exception as e:
        print("Error during prediction:", e)
        return str(e)

# Define the Gradio interface
inputs = [
    gr.Radio(label="Gender", choices=["Female", "Male"]),
    gr.Number(label="Senior Citizen (0 or 1)"),
    gr.Radio(label="Partner", choices=["Yes", "No"]),
    gr.Radio(label="Dependents", choices=["Yes", "No"]),
    gr.Number(label="Tenure (integer)"),
    gr.Radio(label="Phone Service", choices=["Yes", "No"]),
    gr.Radio(label="Multiple Lines", choices=["Yes", "No", "No phone service"]),
    gr.Radio(label="Internet Service", choices=["DSL", "Fiber optic", "No"]),
    gr.Radio(label="Online Security", choices=["Yes", "No", "No internet service"]),
    gr.Radio(label="Online Backup", choices=["Yes", "No", "No internet service"]),
    gr.Radio(label="Device Protection", choices=["Yes", "No", "No internet service"]),
    gr.Radio(label="Tech Support", choices=["Yes", "No", "No internet service"]),
    gr.Radio(label="Streaming TV", choices=["Yes", "No", "No internet service"]),
    gr.Radio(label="Streaming Movies", choices=["Yes", "No", "No internet service"]),
    gr.Radio(label="Contract", choices=["Month-to-month", "One year", "Two year"]),
    gr.Radio(label="Paperless Billing", choices=["Yes", "No"]),
    gr.Radio(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
    gr.Number(label="Monthly Charges (float)"),
    gr.Number(label="Total Charges (float)")
]

outputs = gr.Textbox(label="Prediction")

# Create the Gradio interface
gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Churn Prediction Model").launch()