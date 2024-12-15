
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Define your custom layer class if using custom layers in the model
class ExpandDimsLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=-1)

# Load the model function with custom_objects argument
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            'model1.h5',
            custom_objects={'ExpandDimsLayer': ExpandDimsLayer},
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Transform data function
def Transform_data(data):
    encoding_data = {'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2}
    data_encoded = data.replace(encoding_data)
    x = data_encoded.drop(["Label"], axis=1)
    y = data_encoded['Label'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(x)
    Y = to_categorical(y)
    return X, Y

# Generate Emotion Report
def generate_emotion_report(results_df):
    """
    Generate a detailed report based on emotion predictions
    """
    # Count emotions
    emotion_counts = results_df['Predicted Sentiment'].value_counts()
    total_samples = len(results_df)
    
    # Generate report sections based on emotions
    report = {
        "POSITIVE": {
            "summary": "Positive Emotional State Insights",
            "description": """
            A positive emotional state indicates optimism, pleasure, and engagement. 
            Key characteristics include:
            - High levels of motivation and energy
            - Increased cognitive flexibility
            - Potential for creative problem-solving
            - Enhanced interpersonal interactions
            """,
            "recommendations": [
                "Maintain current positive mental strategies",
                "Continue activities that contribute to positive emotional states",
                "Explore opportunities for personal growth and development"
            ]
        },
        "NEUTRAL": {
            "summary": "Neutral Emotional State Insights",
            "description": """
            A neutral emotional state suggests a balanced and stable mental condition. 
            Characteristics include:
            - Emotional equilibrium
            - Reduced stress and anxiety
            - Potential for objective decision-making
            - Mental clarity and composure
            """,
            "recommendations": [
                "Practice mindfulness to maintain emotional balance",
                "Engage in activities that promote mental well-being",
                "Explore strategies to enhance emotional engagement"
            ]
        },
        "NEGATIVE": {
            "summary": "Negative Emotional State Insights",
            "description": """
            A negative emotional state indicates potential stress, anxiety, or emotional challenge. 
            Key observations include:
            - Potential emotional fatigue
            - Reduced cognitive performance
            - Increased risk of mental health concerns
            - Need for emotional support and intervention
            """,
            "recommendations": [
                "Seek professional mental health support if persistently negative",
                "Practice stress-reduction techniques",
                "Engage in self-care and supportive activities",
                "Consider discussing emotional challenges with trusted individuals"
            ]
        }
    }
    
    # Prepare the report
    st.header("Comprehensive Emotion Detection Report")
    
    # Overall emotion distribution
    st.subheader("Emotion Distribution")
    emotion_col1, emotion_col2 = st.columns(2)
    with emotion_col1:
        st.metric("Total Samples", total_samples)
    with emotion_col2:
        for emotion, count in emotion_counts.items():
            percentage = (count / total_samples) * 100
            st.metric(f"{emotion} Emotion", f"{count} ({percentage:.2f}%)")
    
    # Detailed insights for each detected emotion
    st.subheader("Detailed Emotion Insights")
    for emotion in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
        if emotion in emotion_counts.index:
            with st.expander(f"{emotion} Emotion Insights"):
                report_section = report[emotion]
                st.markdown(f"### {report_section['summary']}")
                st.write(report_section['description'])
                
                st.markdown("#### Recommendations:")
                for rec in report_section['recommendations']:
                    st.markdown(f"- {rec}")

# Map numerical predictions to text labels
label_mapping = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

# Streamlit UI for model prediction
st.set_page_config(page_title="CSV Model Detection", page_icon="📊", layout="wide")

# Sidebar for model and info
st.sidebar.title("Model Detection App")
st.sidebar.write("Upload a CSV file with labeled data, and the model will predict the sentiment of each entry.")
st.sidebar.markdown("---")

# Title and description
st.title("Emotion Detection from Brain Signals")
st.write("""
    This application takes brain signal data from a CSV file and uses a pre-trained model to detect emotion labels:
    - *0:* NEGATIVE
    - *1:* NEUTRAL
    - *2:* POSITIVE
""")

# Load the model once and use it for all predictions
model = load_model()
if model is None:
    st.error("Model failed to load. Please check the model file.")
else:
    # File uploader for CSV input
    st.header("Upload Your Data File")
    csv_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if csv_file:
        testdata = pd.read_csv(csv_file)
        
        if testdata is not None:
            st.subheader("Uploaded Data Preview")
            st.write("Below is a preview of the uploaded data.")
            st.write(testdata.head())  # Show first few rows of the file

            # Process and transform data
            x, y = Transform_data(testdata)

            # Make predictions
            prediction = model.predict(x)
            prediction1 = np.argmax(prediction, axis=1)
            test1 = np.argmax(y, axis=1)

            # Map predictions to labels
            predicted_labels = [label_mapping[pred] for pred in prediction1]
            actual_labels = [label_mapping[actual] for actual in test1]

            # Calculate accuracy
            accuracy = accuracy_score(test1, prediction1)

            # Display results
            st.header("Detection Results")
            st.success(f"Detection Accuracy: {accuracy:.2%}")

            # Display prediction and actual labels for each row
            st.subheader("Sample Detections")
            results_df = pd.DataFrame({
                "Predicted Label (Argmax)": prediction1,
                "Predicted Sentiment": predicted_labels,
                "Actual Label (Argmax)": test1,
                "Actual Sentiment": actual_labels
            })
            st.write(results_df.head())  # Display a few sample predictions

            # Generate and display emotion report
            generate_emotion_report(results_df)

            # Optional: Display the full prediction data as a download link
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediction Results as CSV",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv",
            )

            # Final message
            st.info("Upload a new CSV file to make more predictions.")