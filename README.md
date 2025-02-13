# Advanced Medical Image Analysis System Using GAN

Welcome to the Advanced Medical Image Analysis System, a Streamlit-based application designed to analyze medical images, provide cancer probability predictions, and offer detailed insights using cutting-edge AI technologies.

## Features

- **Image Analysis:** Upload medical images or provide an image URL for analysis.
- **Cancer Probability Prediction:** Analyze the likelihood of cancer using a GAN-based discriminator.
- **Detailed Medical Findings:** Leverages the Gemini API for comprehensive medical image analysis.
- **Medical Chatbot:** Provides context-aware, empathetic, and professional responses using the Groq API.

## Prerequisites

- Python 3.8 or higher
- Required Python packages (see Dependencies)
- Environment variables for API keys:
  - `GEMINI_API_KEY`: Gemini API key
  - `GROQ_API_KEY`: Groq API key

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/soumyajitjalua1/Cancers_prediction_Using_GAN.git
   cd Cancers_prediction_Using_GAN

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Set up environment variables: Create a .env file in the project root directory with the following content:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key




## Application Workflow
1. Choose between Upload Image or Provide Image URL for analysis.
2. View cancer probability predictions and risk categorization.
3. Explore detailed medical findings via Gemini API analysis.
4. Engage with the Medical Consultation Chatbot for further insights.

## Key Components
### Cancer Probability Prediction
- Uses a GAN-based discriminator model to predict cancer probability from preprocessed medical images.
### Gemini API Analysis
- Provides comprehensive and professional medical insights.
- Requires a valid GEMINI_API_KEY.
### Medical Chatbot
- Offers empathetic, context-aware responses to user queries.
- Requires a valid GROQ_API_KEY.

## Dependencies
- TensorFlow
- Streamlit
- NumPy
- OpenCV
- Pillow
- Python-dotenv
- Requests
- Google Generative AI Library
- Groq Library

  ## Install all dependencies via:
  ```bash
  pip install -r requirements.txt

## File Structure
- app.py: Main application script
- requirements.txt: Dependency list
- generator_model.h5: Pre-trained generator model (optional)
- discriminator_model.h5: Pre-trained discriminator model (optional)
 
## Error Handling
- API Key Validation: Verifies presence of required API keys.
- Image Analysis: Handles errors related to image processing and API response issues.
- Chatbot Response: Manages token limits and provides fallback messages for failures.

## Disclaimer
- This application is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.
   ```bash
  Save this as `README.md` in your project directory, and then follow the steps I mentioned earlier to add and push it to your GitHub repository.

## Usage
Run the Streamlit application:
```bash
streamlit run app.py



