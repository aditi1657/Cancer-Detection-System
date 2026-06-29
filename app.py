from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the dataset path
DATA_PATH = "breast_cancer_dataframe.csv"

# Preprocess and train the model
def train_model():
    # Load and clean the dataset
    data = pd.read_csv(DATA_PATH)
    
    # Ensure correct number of columns
    target_column = "target"  # Replace with the correct column name if different
    if target_column not in data.columns:
        raise ValueError("Target column not found in dataset.")

    # Features (X) and Target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.2f}")

    return model, X.columns.tolist()

# Train the model and fetch feature columns
model, feature_columns = train_model()

@app.route('/')
def home():
    return render_template('index.html', feature_columns=feature_columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the user
        input_features = [float(request.form[feature]) for feature in feature_columns]
        
        # Make a prediction
        prediction = model.predict([input_features])[0]
        result = "Breast Cancer Detected" if prediction == 1 else "No Breast Cancer Detected"
        
        return render_template('index.html', feature_columns=feature_columns, prediction_text=f"Prediction: {result}")
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
