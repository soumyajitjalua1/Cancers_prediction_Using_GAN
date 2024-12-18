import streamlit as st
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
import cv2
import os
import PIL.Image
import io
import requests

# AI Libraries
import google.generativeai as genai
from groq import Groq

class CancerPredictionApp:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Setup APIs
        self.setup_apis()

        # Load or create models
        self.load_models()

        # Initialize session state for conversation
        if 'medical_context' not in st.session_state:
            st.session_state.medical_context = ""

    def setup_apis(self):
        """
        Initialize API clients with enhanced error handling
        """
        # Gemini API for image analysis
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not self.GEMINI_API_KEY:
            st.error("Gemini API key is missing. Please set it in your environment variables.")
        else:
            try:
                genai.configure(api_key=self.GEMINI_API_KEY)
                # Use the new Gemini 1.5 Flash model
                self.gemini_vision_model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                st.error(f"Failed to initialize Gemini API: {e}")

        # Groq API for conversational AI
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not self.GROQ_API_KEY:
            st.error("Groq API key is missing. Please set it in your environment variables.")
        else:
            try:
                self.groq_client = Groq(api_key=self.GROQ_API_KEY)
            except Exception as e:
                st.error(f"Failed to initialize Groq API: {e}")

    def load_models(self):
        """
        Load or create default models
        """
        try:
            self.generator = tf.keras.models.load_model('generator_model.h5')
            self.discriminator = tf.keras.models.load_model('discriminator_model.h5')
        except Exception as e:
            st.warning("Creating default models...")
            self.generator, self.discriminator = self._create_default_models()
            self.generator.save('generator_model.h5')
            self.discriminator.save('discriminator_model.h5')

    def _create_default_models(self, img_shape=(256, 256, 1), latent_dim=100):
        """
        Create default GAN models
        """
        from tensorflow import keras
        from tensorflow.keras import layers

        # Generator Model (simplified)
        generator = keras.Sequential([
            layers.Dense(128 * 64 * 64, input_dim=latent_dim),
            layers.Reshape((64, 64, 128)),
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (5, 5), padding='same', activation='tanh')
        ])

        # Discriminator Model (simplified)
        discriminator = keras.Sequential([
            layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', 
                          input_shape=img_shape),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

        discriminator.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return generator, discriminator

    def analyze_image_with_gemini(self, image_path):
        """
        Perform comprehensive medical image analysis with robust error handling
        """
        try:
            # Validate image file
            if not os.path.exists(image_path):
                return "Error: Image file not found."

            # Open image using OpenCV for additional validation
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                return "Error: Unable to read the image. Check file format and integrity."

            # Open image using PIL
            img = PIL.Image.open(image_path)

            # Comprehensive prompting for medical image analysis
            prompt = (
                "You are a professional medical imaging specialist. Perform a detailed analysis "
                "of this medical scan. Provide a comprehensive interpretation including:\n"
                "1. Precise Structural Observations\n"
                "2. Detailed Potential Abnormality Detection\n"
                "3. Specific Recommended Diagnostic Next Steps\n"
                "4. Critical Points for Medical Consultation\n"
                "Use a clear, professional, and compassionate medical communication style. "
                "Emphasize the importance of professional medical verification."
            )

            # Generate analysis with safety settings
            safety_settings = {
                'HARASSMENT': 'BLOCK_NONE',
                'HATE': 'BLOCK_NONE', 
                'SEXUAL': 'BLOCK_NONE',
                'DANGEROUS': 'BLOCK_NONE'
            }

            response = self.gemini_vision_model.generate_content(
                [prompt, img], 
                safety_settings=safety_settings
            )

            # Check if response generation was successful
            if not response or not response.text:
                return (
                    "Unable to generate image analysis. "
                    "Possible reasons:\n"
                    "- Image processing limitations\n"
                    "- Unexpected API response\n\n"
                    "Recommendation: Consult a medical professional for accurate interpretation."
                )

            return response.text

        except Exception as e:
            st.error(f"Detailed Image Analysis Error: {e}")
            return (
                "Unable to perform comprehensive image analysis. "
                "Possible reasons include:\n"
                "- Unsupported image format\n"
                "- Image quality issues\n"
                "- Technical limitations\n\n"
                "Recommendation: Consult a medical professional for accurate interpretation."
            )

    def analyze_image_from_url(self, image_url):
        """
        Perform image analysis from a URL
        """
        try:
            # Download the image from the URL
            response = requests.get(image_url)
            if response.status_code != 200:
                return "Error: Unable to fetch the image from the provided URL."

            img = PIL.Image.open(io.BytesIO(response.content))

            # Save image temporarily for analysis
            temp_path = "temp_url_image.png"
            img.save(temp_path)

            # Perform analysis
            result = self.analyze_image_with_gemini(temp_path)

            # Remove the temporary file
            os.remove(temp_path)

            return result
        except Exception as e:
            st.error(f"Error analyzing image from URL: {e}")
            return "Error: Unable to analyze image from the provided URL."

    def get_medical_chatbot_response(self, user_query):
        """
        Enhanced medical chatbot response with context-aware guidance
        """
        try:
            # Truncate context to manage token limits
            context = self.truncate_context(st.session_state.medical_context)

            # Enhanced system prompt for more empathetic and precise responses
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an advanced medical AI assistant specializing in medical imaging and cancer analysis. "
                        "Provide scientifically accurate, empathetic, and nuanced medical guidance. "
                        "Your responses should:\n"
                        "- Be based on the provided medical context\n"
                        "- Offer clear, actionable insights\n"
                        "- Emphasize the importance of professional medical consultation\n"
                        "- Maintain a supportive and compassionate tone"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Medical Context: {context}\n\n"
                        f"User Question: {user_query}\n\n"
                        "Provide a comprehensive, supportive, and medically informed response. "
                        "Clearly communicate the limitations of AI-based analysis and the critical need for professional medical evaluation."
                    )
                }
            ]

            # Generate response with more thoughtful parameters
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                max_tokens=750,  # Slightly increased for more detailed responses
                temperature=0.7,  # Slightly higher for more nuanced responses
                top_p=0.9  # Allows more diverse token selection
            )

            response = chat_completion.choices[0].message.content

            # Add a disclaimer to the response
            disclaimer = (
                "\n\n---\n"
                "**Disclaimer:** This AI-generated medical information is for informational "
                "purposes only and should not replace professional medical advice, diagnosis, "
                "or treatment. Always consult with a qualified healthcare provider."
            )

            return response + disclaimer

        except Exception as e:
            return (
                f"Chatbot Error: Unable to generate response. {str(e)}\n\n"
                "We recommend consulting directly with a medical professional for personalized advice."
            )

    def truncate_context(self, context, max_tokens=1000):
        """
        Truncate medical context to manage token limits
        """
        # Simple token estimation (1 token ≈ 4 characters)
        if len(context) > max_tokens * 4:
            return context[-max_tokens*4:]
        return context

    def predict_cancer(self, processed_image):
        """
        Predict cancer probability
        """
        try:
            cancer_prob = self.discriminator.predict(processed_image)[0][0]
            return cancer_prob
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

    def preprocess_image(self, uploaded_file):
        """
        Preprocess uploaded image
        """
        # Read image using PIL to handle various formats
        img = PIL.Image.open(uploaded_file)

        # Convert to grayscale
        img_gray = img.convert('L')

        # Resize
        img_resized = img_gray.resize((256, 256))

        # Convert to numpy array and normalize
        img_array = np.array(img_resized)
        img_normalized = img_array / 255.0
        img_normalized = (img_normalized - 0.5) * 2
        img_input = img_normalized.reshape(1, 256, 256, 1)

        return img_input

    def setup_streamlit(self):
        """
        Streamlit application interface with improved error handling
        """
        st.title("🏥 Advanced Medical Image Analysis System")

        # Validation and prerequisites
        if not self.GEMINI_API_KEY:
            st.error("❌ Gemini API key is required for image analysis.")
            return

        if not self.GROQ_API_KEY:
            st.error("❌ Groq API key is required for medical consultation.")
            return

        # Option to upload an image or provide a URL
        analysis_option = st.radio(
            "Choose your input method",
            ("Upload Image", "Provide Image URL")
        )

        image_path = None

        if analysis_option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload Medical Image", 
                type=['png', 'jpg', 'jpeg', 'dcm', 'tif'],
                help="Upload a clear, high-quality medical imaging scan for analysis"
            )

            if uploaded_file is not None:
                # Temporary save for analysis
                image_path = "temp_uploaded_image.png"

                # Save uploaded file
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Display uploaded image
                st.image(uploaded_file, caption="Uploaded Medical Image", use_column_width=True)

        elif analysis_option == "Provide Image URL":
            image_url = st.text_input("Enter the Image URL")

            if image_url:
                st.image(image_url, caption="Image from URL", use_column_width=True)
                analysis_result = self.analyze_image_from_url(image_url)
                st.write("Detailed Medical Findings:")
                st.text(analysis_result)
                return

        if image_path:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)

            # Cancer Probability Prediction
            cancer_probability = self.predict_cancer(processed_image)

            if cancer_probability is not None:
                # Detailed Medical Analysis
                st.subheader("📊 Comprehensive Medical Analysis")

                # Risk Assessment
                st.metric(label="Cancer Probability", value=f"{cancer_probability * 100:.2f}%")

                # Risk Categorization
                if cancer_probability > 0.7:
                    risk_level = "High Risk 🚨"
                    risk_color = "red"
                elif cancer_probability > 0.4:
                    risk_level = "Moderate Risk ⚠️"
                    risk_color = "orange"
                else:
                    risk_level = "Low Risk ✅"
                    risk_color = "green"

                # Detailed Analysis
                try:
                    gemini_analysis = self.analyze_image_with_gemini(image_path)

                    # Display Analysis
                    with st.expander("📝 Detailed Medical Findings"):
                        st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                        st.write(gemini_analysis)

                    # Store medical context for chatbot
                    st.session_state.medical_context = f"""
                    Medical Image Analysis:
                    - Cancer Probability: {cancer_probability * 100:.2f}%
                    - Risk Level: {risk_level}

                    Key Medical Findings:
                    {gemini_analysis}
                    """

                    # Medical Consultation Chatbot
                    st.subheader("💬 Medical Consultation Chatbot")

                    # User query input
                    user_query = st.text_input("Ask a question about your medical analysis")

                    if user_query:
                        if self.GROQ_API_KEY:
                            # Generate chatbot response
                            chatbot_response = self.get_medical_chatbot_response(user_query)

                            # Display chatbot response
                            st.info(chatbot_response)
                        else:
                            st.warning("Groq API key is required for the medical chatbot.")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

                # Remove temporary file
                os.remove(image_path)


def main():
    app = CancerPredictionApp()
    app.setup_streamlit()

if __name__ == "__main__":
    main()
