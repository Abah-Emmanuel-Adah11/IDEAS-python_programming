# Real Estate Location Prediction

## Project Overview
This project aims to predict the best location for purchasing housing property in Nigeria based on budget and other features using machine learning. It is a classification problem where the model predicts the town based on various numerical and categorical features.

## Dataset
The dataset used in this project is sourced from **Kaggle**.

### Features and Target Variable
- **Title (string):** Type of house (e.g., Semi-Detached Duplex, Detached Duplex, Bungalow)
- **Parking Space (int):** Number of parking spaces available in the property
- **Bedrooms (int):** Number of bedrooms
- **Bathrooms (int):** Number of bathrooms
- **Price (float):** Cost of the property
- **Town (string - Target Variable):** Predicted town for the property

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib/Seaborn

## Installation and Setup
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <project-folder>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```sh
   jupyter notebook
   ```

## Usage
- Load the dataset and preprocess it
- Train a machine learning classification model
- Evaluate model performance
- Use the trained model to predict the town based on new property details

## Contributors
- Group 3

## License
This project is licensed under the MIT License.
