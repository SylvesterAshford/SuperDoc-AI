"""
Smart Document Classifier with LLM Analysis
=============================================
A Streamlit web application that:
  1. Accepts document uploads (.txt, .docx, .pdf)
  2. Classifies them into categories using pre-trained ML models
     (Logistic Regression, SVM, Random Forest)
  3. Optionally generates a deeper analysis using an LLM (Llama-3 via Together AI)
  4. Exports a downloadable PDF report of the results
"""

# --- Standard library imports ---
import re          # Regular expressions for text cleaning
import os          # OS-level utilities (not actively used but available)
from io import BytesIO  # In-memory binary stream (used for file exports)

# --- Third-party imports ---
import streamlit as st                # Streamlit framework for building the web UI
import joblib                         # Load pre-trained scikit-learn models/vectorizers
import pandas as pd                   # DataFrames for displaying prediction results
import matplotlib.pyplot as plt       # Plotting (available for future use)
import numpy as np                    # Numerical operations (available for future use)
import seaborn as sns                 # Statistical visualization (available for future use)
import PyPDF2                         # Read and extract text from PDF files
from docx import Document             # Read .docx (Word) files
from fpdf import FPDF                 # Generate PDF report files
from PIL import Image                 # Image handling (available for future use)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # ML metrics (available for evaluation)

# --- LangChain imports for LLM integration ---
from langchain_openai import ChatOpenAI            # OpenAI-compatible chat model wrapper
from langchain_core.prompts import PromptTemplate   # Templated prompts for the LLM


# ──────────────────────────────────────────────
# Streamlit page configuration
# Sets the browser tab title and uses a wide layout for more screen space.
# ──────────────────────────────────────────────
st.set_page_config(page_title="Smart Document Analysis", layout="wide")



def clean_text(text):
    """
    Pre-process raw document text before feeding it to the TF-IDF vectorizer.

    Steps:
      1. Convert all characters to lowercase for consistency.
      2. Remove any character that is not alphanumeric or whitespace,
         stripping punctuation, special symbols, etc.

    Args:
        text (str): The raw text extracted from the uploaded document.

    Returns:
        str: Cleaned text suitable for vectorization.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]","",text)

    return text


def extract_text(uploaded_file):
    """
    Read the uploaded file and extract its text content.

    Supports three file formats:
      - .txt  → decoded directly as UTF-8
      - .docx → paragraphs joined with newlines using python-docx
      - .pdf  → pages extracted and joined with newlines using PyPDF2

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        str | None: The extracted text, or None if the file type is
                     unsupported or an error occurs during reading.
    """
    text = ""

    try:
        if uploaded_file.name.endswith(".txt"):
            # Plain text files — read bytes and decode to string
            text = uploaded_file.read().decode("utf-8")

        elif uploaded_file.name.endswith(".docx"):
            # Word documents — iterate over paragraphs
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])

        elif uploaded_file.name.endswith("pdf"):
            # PDF files — iterate over each page and extract text
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages])
        
        else:
            # Unsupported format — show an error in the Streamlit UI
            st.error("Unsupported file type! Please upload .txt,.docx, or .pdf")
            return None
    except Exception as e:
        # Catch-all for any read errors (corrupt files, encoding issues, etc.)
        st.error(f"Error reading file: {str(e)}")
        return None
    
    return text


@st.cache_resource  # Cache loaded models/vectorizer across reruns to avoid reloading on every interaction
def load_assets():
    """
    Load and cache the TF-IDF vectorizer and all pre-trained classification models.

    The vectorizer transforms raw text into numerical feature vectors.
    Three models are loaded:
      - Logistic Regression
      - Support Vector Machine (SVM)
      - Random Forest

    Returns:
        dict: A dictionary with two keys:
              'vectorizer' → the fitted TF-IDF vectorizer
              'models'     → a dict mapping model names to trained model objects
    """
    return {
        'vectorizer':joblib.load("notebooks/tfidf_vectorizer.joblib"),
        'models' : {
            "Logistic Regression":joblib.load("models/logistic_regression.pkl"),
            "SVM": joblib.load("models/svm.pkl"),
            "Random Forest": joblib.load("models/random_forest.pkl")
        }

    }
    

def generate_llm_analysis(text, category, tempeature=0.7):
    """
    Send the document text and its predicted category to an LLM for deeper analysis.

    Uses the Llama-3-8b model hosted on Together AI (OpenAI-compatible endpoint).
    The LLM is asked to:
      1. Generate a suitable title for the document.
      2. Explain why the title fits.
      3. Explain why the predicted category is appropriate.

    Args:
        text (str):        The raw (uncleaned) document text.
        category (str):    The predicted category from the ML model.
        tempeature (float): Controls randomness/creativity of the LLM output.
                            Lower = more deterministic, higher = more creative.

    Returns:
        str: The LLM's generated analysis text.
    """
    # Initialize the LLM client pointing to Together AI's API
    llm = ChatOpenAI(
        model="meta-llama/Llama-3-8b-chat-hf",
        temperature=tempeature,
        api_key=st.secrets["TOGETHER_API_KEY"],  # API key stored in Streamlit secrets
        base_url="https://api.together.xyz/v1",
    )

    # Define a prompt template with placeholders for the document text and category
    prompt = PromptTemplate.from_template("""
    Document Analysis:
    {text}
    
    Predicted Category: {category}
    
    1. Generate a suitable title
    2. Explain why this title fits
    3. Explain why the {category} category is appropriate
    """)
    
    # Invoke the LLM with the formatted prompt
    # Text is truncated to 2000 chars to stay within token limits
    response = llm.invoke(prompt.format(
        text=text[:2000],  # Limit text length to avoid exceeding context window
        category=category
    ))
    
    return response.content

def create_pdf_report(predictions, llm_analysis=None):
    """
    Generate a PDF report containing model predictions and optional LLM analysis.

    The report includes:
      - A centered title header
      - A list of each model's prediction and confidence score
      - (Optional) The LLM-generated analysis section

    Args:
        predictions (list[dict]): A list of dicts with keys 'Model', 'Prediction', 'Confidence'.
        llm_analysis (str | None): The LLM's analysis text, or None to omit that section.

    Returns:
        bytes: The PDF file content as a byte string, ready for download.
    """
    pdf = FPDF ()
    pdf.add_page()
    pdf.set_font("Arial", size =14)


    # Title — centered at the top of the page
    pdf.cell(200, 10, txt="Document Classification Report", ln=1, align='C')
    pdf.ln(10)  # Add vertical spacing

    # Model predictions section header (bold)
    pdf.set_font("Arial", size =12, style="B")
    pdf.cell(200, 10, txt="Model Predictions:", ln=1)
    pdf.set_font("Arial", size=10)

    # List each model's prediction and confidence on its own line
    for pred in predictions:
        pdf.cell(200, 10, 
                txt=f"{pred['Model']}: {pred['Prediction']} ({pred['Confidence']})", 
                ln=1)
    
    # LLM Analysis section — only included if analysis text was provided
    if llm_analysis:
        pdf.ln(10)  # Add spacing before the section
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(200, 10, txt="LLM Analysis:", ln=1)
        pdf.set_font("Arial", size=10)
        # multi_cell automatically wraps long text across multiple lines
        pdf.multi_cell(0, 10, txt=llm_analysis)
    
    # Output the PDF as a byte string (latin1 encoding required by FPDF)
    return pdf.output(dest='S').encode('latin1')


def main():
    """
    Main application entry point — builds the Streamlit UI and orchestrates
    the full document classification and analysis workflow.

    Flow:
      1. User uploads a document via the file uploader widget.
      2. Raw text is extracted and cleaned.
      3. A preview of the document is shown in an expandable section.
      4. The text is vectorized and classified by all three ML models.
      5. Predictions (with confidence scores) are displayed in a table.
      6. The best prediction (highest confidence) is highlighted.
      7. User can adjust LLM temperature and choose which model's prediction
         to send to the LLM for deeper analysis.
      8. On clicking "Generate Analysis", the LLM produces a title + explanation.
      9. A downloadable PDF report is generated with all results.
    """
    # Page title displayed at the top of the app
    st.title("Smart Document Classifier with LLM Analysis")

    # File upload widget — accepts .txt, .docx, and .pdf files
    uploaded_file = st.file_uploader("Upload document", type=['txt','docx','pdf'])

    if uploaded_file:
        # Step 1: Extract raw text from the uploaded file
        raw_text = extract_text(uploaded_file)
        # Step 2: Clean the text (lowercase, remove special chars) for ML models
        cleaned_text = clean_text(raw_text)


        # Show a collapsible preview of the first 500 characters of the document
        with st.expander("📄 Document Preview"):
            st.text(raw_text[:500]+ "..."  if len(raw_text)>500 else raw_text)


        # Step 3: Load cached models and vectorize the cleaned text
        assets = load_assets()
        text_vec = assets['vectorizer'].transform([cleaned_text])  # TF-IDF transform


        # Step 4: Run predictions with all three models
        st.subheader("Model Prediction")
        result= []

        for name, model in assets["models"].items():
            prediction = model.predict(text_vec)[0]         # Get the predicted category label
            prob = max(model.predict_proba(text_vec)[0])     # Get the highest class probability as confidence
            result.append(
                {
                    "Model":name,
                    "Prediction":prediction,
                    "Confidence":f"{prob:.2%}"               # Format as percentage (e.g., "93.45%")
                }
            )

        # Display predictions in a table
        result_df = pd.DataFrame(result)
        st.dataframe(result_df)

        # Highlight the model with the highest confidence score
        best_result = max(result, key=lambda x : x["Confidence"])
        st.markdown(f"### 🤖 Best Prediction: **{best_result['Prediction']}** by {best_result['Model']} ({best_result['Confidence']} confidence)")

        # --- LLM Analysis Section ---
        st.subheader("🤖 LLM Analysis")

        # Slider to control the LLM's creativity/randomness
        temp = st.slider("LLM Creativity (temperature)", min_value=0.1, max_value=1.0, value=0.7, step=0.1)


        # Let the user pick which model's prediction to use for the LLM prompt
        selected_model = st.selectbox("Use prediction from: ",
        [res["Model"] for res in result])

        # Find the prediction string for the selected model
        selected_pred = next(res["Prediction"] for res in result if res["Model"]==selected_model)

        # Trigger LLM analysis when the button is clicked
        if st.button("Generate Analysis"):
            with st.spinner("Generating LLM Analysis..."):
                try:
                    # Call the LLM with the raw text, selected prediction, and temperature
                    response = generate_llm_analysis(raw_text, selected_pred, temp)

                    # Parse the response — first line as title, rest as analysis body
                    title = response.split("\n")[0]
                    analysis = "\n".join(response.split("\n")[1:])

                    # Display the generated title and analysis
                    st.write("### Generated Title:")
                    st.write(title)
                    st.write("###Analysis")
                    st.write(analysis)

                    # Generate a downloadable PDF report with predictions + LLM analysis
                    pdf_report = create_pdf_report(result,llm_analysis=analysis)
                    st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_report,
                    file_name="classification_report.pdf",
                    mime="application/pdf"
                )
                except Exception as e:
                    # Show error in the UI if LLM call or PDF generation fails
                    st.error(f"LLM analysis failed: {str(e)}")


# Standard Python entry point — runs the main function when the script is executed directly
if __name__ == "__main__":
    main()
