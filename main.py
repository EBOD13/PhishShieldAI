import json
import time
from flask import Flask, request, jsonify, Response
import torch
import os
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)

# VirusTotal API Key (Set your actual API key here)
VIRUSTOTAL_API_KEY = "your_virustotal_api_key"

# Global variable to store scan status
scan_status = {
    "status": "Idle",
    "progress": 0,
    "timestamp": None,
    "duration": None,
    "classification": None,
    "score": None,
}

# Check if the domain is legitimate


# Load the tokenizer and model from the repository, specifying the subfolder
repo_id = "ebod13/phish-email-models"
subfolder = "models/fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
# Load model with memory optimizations
model = AutoModelForSequenceClassification.from_pretrained(
    "ebod13/phish-email-models",
    subfolder="models/fine_tuned_model",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(
    "cpu"
)  # Explicitly move to CPU

# Optionally, move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def classify_email(text):
    # Prepare inputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

    # Map numeric label to custom label
    label_map = {1: "Phish", 0: "Safe", 2: "Spam", 3: "Ham"}
    final_label = label_map.get(predicted_class, "Unknown")

    return final_label, confidence


# Function to perform the scan and update status
def perform_scan(data):
    global scan_status
    start_time = time.time()

    # Update status to "In Progress"
    scan_status["status"] = "In Progress"
    scan_status["progress"] = 0
    scan_status["timestamp"] = datetime.now().strftime("%H:%M%p")
    scan_status["classification"] = None
    scan_status["score"] = None

    # Simulate domain check
    time.sleep(1)
    scan_status["progress"] = 25

    # Simulate tokenization
    time.sleep(1)
    scan_status["progress"] = 50

    # Build the full email text
    email_text = f"Subject: {data['subject']}\nSender: {data['sender']}\n{data['text']}"

    # Use the classify_email function (which uses your new model)
    final_label, confidence = classify_email(email_text)

    # Update scan status with the model prediction
    scan_status["classification"] = final_label
    scan_status["score"] = round(confidence * 100, 2)  # Confidence as a percentage

    time.sleep(1)
    scan_status["progress"] = 75

    # Finalize scan
    scan_status["status"] = "Complete"
    scan_status["progress"] = 100
    scan_status["duration"] = round(time.time() - start_time, 2)


# Main route
@app.route("/")
def home():
    return "PhishShield AI Running"


# SSE endpoint to stream scan status
@app.route("/scan_status")
def scan_status_stream():
    def event_stream():
        global scan_status
        while True:
            yield f"data: {json.dumps(scan_status)}\n\n"
            time.sleep(1)  # Send updates every second

    return Response(event_stream(), mimetype="text/event-stream")


# Quick Scan endpoint
@app.route("/quick_scan", methods=["POST"])
def quick_scan():
    data = request.get_json()
    if not data or "text" not in data or "subject" not in data or "sender" not in data:
        return jsonify({"error": "Missing required fields"}), 400

    # Start the scan in a separate thread
    threading.Thread(target=perform_scan, args=(data,)).start()

    return jsonify({"message": "Scan started"})


# Run Flask Server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Heroku's port or default to 5000
    app.run(debug=True, host="0.0.0.0", port=port)
