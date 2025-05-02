import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load disease-symptom data
try:
    symptom_data = pd.read_csv('data/disease_symptom.csv')
    drug_data = pd.read_csv('data/disease_drug.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

if "Disease" not in symptom_data.columns or "Disease" not in drug_data.columns:
    print("Error: 'Disease' column missing from one or both CSV files.")
    exit()

symptom_diseases = set(symptom_data["Disease"].unique())
drug_diseases = set(drug_data["Disease"].unique())

if not symptom_diseases.issubset(drug_diseases):
    print("Error: there are diseases in the symptom file that are not in the drug file.")
    exit()

symptoms = pd.get_dummies(symptom_data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4']].stack()).groupby(level=0).sum()
diseases = symptom_data['Disease']

# Load disease-drug data
drugs = drug_data.set_index('Disease')['Drugs']  # set index to disease

# Encode diseases
label_encoder = LabelEncoder()
encoded_diseases = label_encoder.fit_transform(diseases)

# Split data
X_train, X_test, y_train, y_test = train_test_split(symptoms, encoded_diseases, test_size=0.2, random_state=42)
symptom_columns = symptoms.columns

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

# Save models and data
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/symptom_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('models/drugs.pkl', 'wb') as f:
    pickle.dump(drugs, f)
with open('models/symptom_columns.pkl', 'wb') as f:
    pickle.dump(symptom_columns, f)