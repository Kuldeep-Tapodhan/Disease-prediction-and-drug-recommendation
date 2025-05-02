import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import pickle
import os

# Load dataset
df = pd.read_csv("../data/dataset.csv")

# Preprocess symptoms
for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5']:
    df[col] = df[col].str.lower().str.strip()

# Drop unnecessary columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
y = df['Disease']
X = df.drop('Disease', axis=1)

# Impute missing symptoms
def impute_missing_symptoms(X_train, X_test):
    imputer = SimpleImputer(strategy="constant", fill_value="unknown")
    filled_X_train = imputer.fit_transform(X_train)
    filled_X_test = imputer.transform(X_test)

    return pd.DataFrame(filled_X_train, columns=X_train.columns, index=X_train.index), \
           pd.DataFrame(filled_X_test, columns=X_test.columns, index=X_test.index)

# K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Impute missing values
    X_train, X_test = impute_missing_symptoms(X_train, X_test)

    # One-hot encoding
    categorical_features = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4", "Symptom_5"]
    one_hot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

    transformed_X_train = transformer.fit_transform(X_train)
    transformed_X_test = transformer.transform(X_test)

    # Convert to DataFrame
    transformed_X_train = pd.DataFrame(transformed_X_train, columns=transformer.get_feature_names_out())
    transformed_X_test = pd.DataFrame(transformed_X_test, columns=transformer.get_feature_names_out())

    # Train the model
    clf = RandomForestClassifier(random_state=42, max_depth=6, class_weight="balanced")
    clf.fit(transformed_X_train, y_train)

    # Predictions
    y_pred = clf.predict(transformed_X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    accuracies.append(accuracy)

# Save the model
model_data = {'model': clf, 'transformer': transformer}
with open('../models/train_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

# Prediction example
input_data = pd.DataFrame({
    'Symptom_1': ['itching'],
    'Symptom_2': ['skin_rash'],
    'Symptom_3': ['nodal_skin_eruptions'],
    'Symptom_4': ['dischromic_patches'],
    'Symptom_5': ['dischromic_patches']
})

# Load the model
if os.path.exists('../models/train_model.pkl'):
    with open('../models/train_model.pkl', 'rb') as file:
        model_data = pickle.load(file)

    clf = model_data['model']
    transformer = model_data['transformer']

    # Transform input data
    try:
        transformed_input = transformer.transform(input_data)
        transformed_input_df = pd.DataFrame(transformed_input, columns=transformer.get_feature_names_out())
    except ValueError as e:
        print(f"Error transforming input: {e}")
        exit()

    # Make prediction
    try:
        prediction = clf.predict(transformed_input_df)
        predicted_disease = prediction[0]
        probabilities = clf.predict_proba(transformed_input_df)
        confidence = probabilities[0][clf.classes_.tolist().index(predicted_disease)]
    except Exception as e:
        print(f"Error making prediction: {e}")
        exit()

    # Output results
    print("Symptom Summary:")
    for i in range(1, 6):
        print(f"- Symptom {i}: {input_data[f'Symptom_{i}'][0]}")
    print(f"\nPredicted Disease: {predicted_disease}")
    print(f"Prediction Confidence: {confidence:.4f}")

    print("\nDisclaimer:")
    print("This prediction is based on limited symptom data and should not be used for medical diagnosis.")
    print("Consult a healthcare professional for accurate diagnosis and treatment.")
else:
    print("Model file not found. Please ensure the model is trained and saved correctly.")