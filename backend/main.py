from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import pickle
import pandas as pd

app = Flask(__name__, template_folder='../template') #template folder is root

# Load machine learning models and data
try:
    with open('../models/symptom_disease_model.pkl', 'rb') as f:
        ml_model = pickle.load(f)
    with open('../models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('../models/drugs.pkl', 'rb') as f:
        drugs = pickle.load(f)
    with open('../models/symptom_columns.pkl', 'rb') as f:
        symptom_columns = pickle.load(f)
except FileNotFoundError:
    ml_model = None
    label_encoder = None
    drugs = None
    symptom_columns = None

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-pro')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict_disease', methods=['POST'])
def predict_disease():
    if ml_model is None or symptom_columns is None:
        return jsonify({"error": "ML model not available"})
    try:
        input_symptoms = request.json['input'].split(',')
        symptoms_df = pd.DataFrame([input_symptoms], columns=symptom_columns)
        symptoms_df = symptoms_df.reindex(columns=symptom_columns, fill_value=0)
        for symptom in input_symptoms:
            if symptom in symptom_columns:
                symptoms_df[symptom] = 1

        prediction = ml_model.predict(symptoms_df)[0]
        predicted_disease = label_encoder.inverse_transform([prediction])[0]
        recommended_drug = drugs[predicted_disease]
        return jsonify({"disease": predicted_disease, "drug": recommended_drug})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    user_input = request.json['input']
    prompt = f"Based on the following input, provide a helpful medical recommendation: {user_input}"
    try:
        response = gemini_model.generate_content(prompt)
        return jsonify({"recommendation": response.text})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)