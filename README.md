# PhishShieldAI  
**AI-Powered Browser Extension for Real-Time Phishing Detection**  

PhishShieldAI is an innovative browser extension that leverages the power of AI to classify emails as either **phishing** or **safe**. Built using a fine-tuned AI model trained on a comprehensive dataset of phishing emails (including Enron, CEAS-08, TREC-07, and TREC-08), PhishShieldAI significantly improves phishing detection accuracy compared to traditional methods.  

## Key Features  
- **Real-Time Email Classification**: Instantly detects phishing emails, reducing the workload on SOC teams and preventing cyber threats.  
- **Fine-Tuned AI Model**: Utilizes a fine-tuned version of **DistilBERT-base-uncased**, achieving **85% accuracy** in phishing detection.  
- **Future Enhancements**:  
  - **Threat Analysis**: Extract links, images, and files from emails for real-time threat scanning using tools like **VirusTotal** and **URLHaus**.  
  - **Malware Analysis**: Integrate **Cuckoo Sandbox** for automated malware behavior analysis.  
  - **IP Blacklisting**: Allow users and administrators to blacklist sender IPs, improving firewall configurations and strengthening IDS/IPS systems.  

## How It Works  
1. **Dataset Aggregation**: Curated datasets from Enron, CEAS-08, TREC-05, TREC-06, and TREC-07 were combined to train the model.  
2. **Model Fine-Tuning**: Initial attempts with **BERT** and **Random Forest** yielded suboptimal results. Fine-tuning **DistilBERT-base-uncased** on the aggregated dataset significantly improved accuracy.  
3. **Real-Time Detection**: The extension analyzes emails in real time, providing users with instant feedback on potential threats.  

## Future Goals  
- Expand threat analysis capabilities to include **QR codes** and other malicious elements.  
- Enable automated reporting and blacklisting to enhance organizational security.  
- Continuously improve the model with new datasets and advanced AI techniques.  

## Why PhishShieldAI?  
PhishShieldAI is designed to empower users and organizations by providing a proactive defense against phishing attacks. By combining cutting-edge AI with real-time threat analysis, it ensures business continuity 
