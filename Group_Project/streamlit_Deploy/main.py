import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model and necessary objects
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
logistic_classifier = joblib.load("rf_classifier.pkl")
X_train_columns = joblib.load("X_train_columns.pkl")
trimmed_data = joblib.load("trimmed_data.pkl")

# Streamlit App Title
st.title("🏡 Nigerian House Town Predictor")
st.write("Enter house features to predict the most likely town.")

# User Input Fields
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
toilets = st.number_input("Number of Toilets", min_value=1, max_value=10, step=1)
parking_space = st.number_input("Parking Space", min_value=0, max_value=10, step=1)
title = st.selectbox("Property Title", ["Detached Duplex", "Semi-Detached Duplex", "Terraced Duplex", "Bungalow", "Penthouse"])
state = st.selectbox("State", ["Lagos", "Abuja", "Imo", "Rivers", "Ogun", "Kaduna"])
price_millions = st.number_input("Price (in millions)", min_value=1.0, step=0.1)

# Prediction Function
def predict_town(bedrooms, bathrooms, toilets, parking_space, title, state, price_millions):
    try:
        input_data = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'toilets': [toilets],
            'parking_space': [parking_space],
            'title': [title],
            'state': [state],
            'price_millions': [price_millions]
        })

        input_data = pd.get_dummies(input_data, columns=['title', 'state'], drop_first=True)
        input_data = input_data.reindex(columns=X_train_columns, fill_value=0)

        input_data_scaled = scaler.transform(input_data)

        predicted_town_encoded = logistic_classifier.predict(input_data_scaled)[0]
        state_towns = trimmed_data[trimmed_data['state'] == state]['town'].unique()

        if len(state_towns) == 0:
            return "No towns found for the given state in the training data."

        predicted_town = label_encoder.inverse_transform([predicted_town_encoded])[0]

        if predicted_town in state_towns:
            return predicted_town
        else:
            state_encoded_towns = label_encoder.transform(state_towns)
            differences = np.abs(state_encoded_towns - predicted_town_encoded)
            closest_index = np.argmin(differences)
            return state_towns[closest_index]

    except KeyError as e:
        return f"Error: Invalid input feature or state. Check your input. {e}"
    except ValueError as e:
        return f"Error: Invalid input value or data mismatch. {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Predict Button
if st.button("Recommend Town"):
    result = predict_town(bedrooms, bathrooms, toilets, parking_space, title, state, price_millions)
    
    if "Error" in result:
        st.error(f"❌ {result}")
    else:
        # Use a Markdown box to display the result
        st.markdown(f"""
            <div style="background-color: #e2f3e6; padding: 10px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h2 style="color: #2f7d4d; text-align: center;">
                    🏙️ Recommended Town: <span style="font-weight: bold; color: #ff6347;">{result}</span>
                </h2>
            </div>
        """, unsafe_allow_html=True)