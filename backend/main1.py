from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import pickle
import pandas as pd
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Safely get API key from environment
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment variables.")

os.environ['_BARD_API_KEY'] = GEMINI_API_KEY

# App initialization (ensure template folder matches directory)
app = Flask(__name__, template_folder='template')

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Utility function to build project-relative paths
def project_path(*parts):
    return os.path.abspath(os.path.join(app.root_path, *parts))

# Load ML model and transformer
clf = None
transformer = None
model_path = project_path("../models/train_model.pkl")
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    clf = model_data['model']
    transformer = model_data['transformer']
    print("ML model loaded successfully")
else:
    print(f"Model file not found at {model_path}")

# Load drug dataset
df_drug_review = None
disease_drug_dict = {}
drug_data_path = project_path("data/disease_consolidated_treatment_dataset.csv")
try:
    df_drug_review = pd.read_csv(drug_data_path)
    disease_drug_dict = dict(zip(df_drug_review['Condition'], df_drug_review['Drug Name']))
    print("Drug dataset loaded successfully")
except FileNotFoundError:
    print(f"{drug_data_path} not found.")

# Load unique symptoms list
unique_symptoms = []
symptoms_path = project_path("../data/unique_values.pkl")
try:
    with open(symptoms_path, "rb") as f:
        unique_symptoms = pickle.load(f)
except Exception as e:
    print(f"Error loading symptoms list: {e}")

def clean_response(text):
    # Remove markdown formatting
    text = text.replace('**', '').replace('*', '')
    # Convert bullet points to consistent format
    text = text.replace('• ', '• ').replace('* ', '• ')
    # Convert numbered sections to bold
    text = re.sub(r'(\d+\.\s+)([^\n]+)', r'\1<b>\2</b>', text)
    return text

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', symptoms=unique_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    if clf is None or transformer is None:
        return jsonify({'error': 'ML model not available.'})

    selected_symptoms = request.form.getlist('symptoms')
    if not selected_symptoms:
        return jsonify({'error': 'Please select symptoms.'})

    try:
        symptoms = [symptom.strip().lower().replace(" ", "") for symptom in selected_symptoms]
        symptoms = (symptoms + [None] * 5)[:5]  # Ensure exactly 5 symptoms

        input_data = pd.DataFrame({
            'Symptom_1': [symptoms[0]],
            'Symptom_2': [symptoms[1]],
            'Symptom_3': [symptoms[2]],
            'Symptom_4': [symptoms[3]],
            'Symptom_5': [symptoms[4]]
        })

        transformed_input = transformer.transform(input_data)
        transformed_input_df = pd.DataFrame(transformed_input, columns=transformer.get_feature_names_out())
        prediction = clf.predict(transformed_input_df)
        predicted_disease = prediction[0]

        probabilities = clf.predict_proba(transformed_input_df)
        confidence = probabilities[0][clf.classes_.tolist().index(predicted_disease)]

        # Drug recommendation based on predicted disease
        recommended_drugs = []
        if df_drug_review is not None:
            matched_drugs = df_drug_review[df_drug_review['Condition'].str.lower() == predicted_disease.lower()]
            if not matched_drugs.empty:
                top_drugs = matched_drugs['Drug Name'].value_counts().head(3).index.tolist()
                recommended_drugs = top_drugs

        if not recommended_drugs:
            recommended_drug = "Consult a doctor for medication advice"
        else:
            recommended_drug = ", ".join(recommended_drugs) + " (Consult doctor before use)"

        # Generate description using Gemini
        prompt = f"""Provide a detailed medical description of {predicted_disease} with the following sections:
        1. Overview (2-3 sentences)
        2. Common Symptoms (4 Sentences)
        3. Possible Causes (4 Sentences)
        4. General Advice (4 Sentences)
        
        Separate each section with a blank line and use clear formatting."""
        response = model.generate_content(prompt)
        disease_info = clean_response(response.text)

        # Convert line breaks to HTML <br> tags for display
        disease_info = disease_info.replace('\n\n', '<br><br>').replace('\n', '<br>')

        return jsonify({
            'predicted_disease': predicted_disease,
            'recommended_drug': recommended_drug,
            'disease_info': disease_info
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=off)
