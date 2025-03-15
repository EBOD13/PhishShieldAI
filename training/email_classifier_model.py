import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy
from collections import Counter
import joblib


# Function to extract domain from sender
def extract_domain(sender):
    email_match = re.search(r'<([^>]+)>', sender)  # Fixed regex to capture email
    if email_match:
        email = email_match.group(1)
        domain = email.split('@')[-1].lower()
        return domain
    return None


# Function to clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.lower()  # Lowercase the text
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text


# Function to calculate domain entropy
def calculate_entropy(domain):
    counter = Counter(domain)
    total = sum(counter.values())
    probabilities = [count / total for count in counter.values()]
    return entropy(probabilities)


# Function to extract TLD
def extract_tld(domain):
    return domain.split('.')[-1]


# Load datasets
# dataset1 = pd.read_csv("/content/drive/MyDrive/datasets/phish_dataset1.csv") # Uncomment for
# dataset2 = pd.read_csv("/content/drive/MyDrive/datasets/phish_dataset3.csv")
# dataset3 = pd.read_csv("/content/drive/MyDrive/datasets/phish_dataset4.csv")


dataset1 = pd.read_csv("datasets/phish_dataset1.csv")
dataset2 = pd.read_csv("datasets/phish_dataset3.csv")
dataset3 = pd.read_csv("datasets/phish_dataset4.csv")

# Add a source column to identify the dataset
dataset1['source'] = 'dataset1'
dataset2['source'] = 'dataset2'
dataset3['source'] = 'dataset3'

# Combine datasets
combined_data = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)

# Rename columns
combined_data.rename(columns={'body': 'text', 'sender': 'domain'}, inplace=True)

# Convert the 'text' column to string type, handling potential missing values
combined_data['text'] = combined_data['text'].astype(str).fillna('')
combined_data['text'] = combined_data['text'].apply(clean_text)

# Drop rows with missing domains or text
combined_data.dropna(subset=['domain', 'text'], inplace=True)

# Add domain features
combined_data['domain_length'] = combined_data['domain'].apply(len)
combined_data['domain_entropy'] = combined_data['domain'].apply(calculate_entropy)
combined_data['tld'] = combined_data['domain'].apply(extract_tld)

# Encode categorical features
le = LabelEncoder()
combined_data['tld'] = le.fit_transform(combined_data['tld'])

# Initialize and fit the TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
tfidf.fit(combined_data['text'])

# Add TF-IDF features to the dataset
tfidf_features = tfidf.transform(combined_data['text']).toarray()
tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf.get_feature_names_out())

# Combine domain features and TF-IDF features
X = pd.concat([combined_data[['domain_length', 'domain_entropy', 'tld']], tfidf_df], axis=1)
y = combined_data['label']

# Convert all columns to numeric, coercing errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN values
X = X.dropna()
y = y[X.index]  # Align y with the dropped rows in X

# Split the datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model, TF-IDF vectorizer, and LabelEncoder
joblib.dump(model, "models/email_phishing_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")


# Function to predict email
def predict_email(domain, text):
    # Preprocess the input
    domain_length = len(domain)
    domain_entropy = calculate_entropy(domain)
    tld = extract_tld(domain)
    cleaned_text = clean_text(text)

    # Apply TF-IDF transformation to the cleaned text
    tfidf_features = tfidf.transform([cleaned_text]).toarray()

    # Encode the TLD using the same LabelEncoder used during training
    try:
        tld_encoded = le.transform([tld])[0]
    except ValueError:
        tld_encoded = -1  # Default value for unseen TLDs

    # Combine features into a list
    features = [domain_length, domain_entropy, tld_encoded] + list(tfidf_features[0])

    # Create a DataFrame with the correct column names (matching training datasets)
    feature_names = ['domain_length', 'domain_entropy', 'tld'] + list(tfidf.get_feature_names_out())
    features_df = pd.DataFrame([features], columns=feature_names)

    # Ensure all features are numeric and handle missing values
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Predict using the trained model
    prediction = model.predict(features_df)
    return "Phish" if prediction[0] == 1 else "Safe"


"""
# EXAMPLE USAGE
domain1 = "suspicious-domain.com"
text1 = "Dear user, your account has been compromised. Click this link to update your password immediately: http://malicious-link.com"
prediction1 = predict_email(domain1, text1)
print(f"Prediction for example 1: {prediction1}")

domain2 = "support@google.com"
text2 = 
""Dear Customer,

We are writing to inform you about an important update to your account.
As part of our ongoing efforts to enhance security, we have implemented new measures to protect your information.
No action is required from you at this time.

If you have any questions, please contact our support team at 1-800-123-4567.

Thank you for choosing YourBank.

Sincerely,
Chase Bank Support Team  ""

prediction2 = predict_email(domain2, text2)
print(f"Prediction for example 2: {prediction2}")
"""