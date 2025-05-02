from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__, template_folder='../templates')  # Adjust the path to the templates folder

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

# Load machine learning models and data
clf = None
transformer = None
model_path = os.path.join("models", 'train_model.pkl')
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    clf = model_data['model']
    transformer = model_data['transformer']

# Load drug dataset
df_drug_review = pd.read_csv("../data/drug_review_10k_realistic.csv")
disease_drug_dict = dict(zip(df_drug_review['Condition'], df_drug_review['Drug Name']))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', [])

    if not clf or not transformer:
        return jsonify({'error': 'ML model not available.'}), 500

    if len(symptoms) == 0:
        return jsonify({'error': 'Please enter symptoms.'}), 400

    # Ensure the symptoms list has exactly 5 elements
    symptoms = (symptoms + [None] * 5)[:5]

    # Create DataFrame with the correct number of symptoms
    input_data = pd.DataFrame({
        'Symptom_1': [symptoms[0]],
        'Symptom_2': [symptoms[1]],
        'Symptom_3': [symptoms[2]],
        'Symptom_4': [symptoms[3]],
        'Symptom_5': [symptoms[4]]
    })

    # Transform the input data
    transformed_input = transformer.transform(input_data)
    transformed_input_df = pd.DataFrame(transformed_input, columns=transformer.get_feature_names_out())

    # Make predictions
    prediction = clf.predict(transformed_input_df)
    predicted_disease = prediction[0]

    probabilities = clf.predict_proba(transformed_input_df)
    confidence = probabilities[0][clf.classes_.tolist().index(predicted_disease)]

    # Get drug recommendation
    recommended_drug = disease_drug_dict.get(predicted_disease, "No recommendation available.")

    # Use Gemini for description and advice
    prompt = f"""
    Provide a short description of {predicted_disease}, including common symptoms, and offer general advice.
    Important: Emphasize that this is not medical advice and users should consult healthcare professionals.
    """
    response = model.generate_content(prompt)
    description = response.text

    return jsonify({
        'predictedDisease': predicted_disease,
        'confidence': confidence,
        'recommendedDrug': recommended_drug,
        'description': description
    })

if __name__ == '__main__':
    app.run(debug=True)
    