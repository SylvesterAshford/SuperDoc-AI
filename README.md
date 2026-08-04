# SuperDocAI: Learn the Codebase

This guide explains the linked [SmartDocAI repository](https://github.com/yoon-thiri04/Smart-doc-ai) as a learning project. It covers what the app does, how the files fit together, the machine-learning concepts behind it, how to run it, and what to improve next.

## 1. What this project does

SmartDocAI is a Streamlit web app that accepts a TXT, DOCX, or PDF document. It extracts the text, cleans it, converts it into TF-IDF features, sends those features to three saved scikit-learn classifiers, and displays the predicted category. The user can then ask a Llama model, through Together AI and LangChain, to generate a title and explanation and download a PDF report.

The app has two different AI stages:

1. Classical machine learning predicts the category.
2. An LLM creates a human-readable title and explanation.

The LLM does not currently make the category prediction.

The checked-in BBC dataset has 2,225 articles and five labels:

| Label | Articles |
|---|---:|
| sport | 511 |
| business | 510 |
| politics | 417 |
| tech | 401 |
| entertainment | 386 |

The README mentions examples such as Medical and Finance, but those labels are not in the current dataset.

## 2. The data flow

~~~text
Upload TXT / DOCX / PDF
          |
          v
     extract_text
          |
          v
       clean_text
          |
          v
Saved TF-IDF vectorizer
          |
          v
   Sparse feature vector
      /       |        \
     v        v         v
Logistic    SVM    Random Forest
Regression
      \       |        /
       v      v       v
 Predictions and probabilities
             |
             v
      Streamlit results table
             |
       User chooses a model
             |
             v
 Together AI through LangChain
             |
             v
 Title, explanation, and PDF
~~~

The app does not train models when a user uploads a file. Training happened earlier in notebooks/train_models.ipynb. The app loads the trained model files from models/ and the fitted vectorizer from notebooks/.

## 3. Repository map

| Path | Purpose | Learning order |
|---|---|---|
| [app.py](https://github.com/yoon-thiri04/Smart-doc-ai/blob/master/app.py) | Streamlit UI, file parsing, inference, LLM call, and PDF creation | 1 |
| [requirements.txt](https://github.com/yoon-thiri04/Smart-doc-ai/blob/master/requirements.txt) | Python packages | 2 |
| [data/bbc-text.csv](https://github.com/yoon-thiri04/Smart-doc-ai/blob/master/data/bbc-text.csv) | Training data with category and text columns | 3 |
| [notebooks/train_models.ipynb](https://github.com/yoon-thiri04/Smart-doc-ai/blob/master/notebooks/train_models.ipynb) | Text cleaning, TF-IDF, training, saving, and evaluation | 4 |
| [notebooks/test_predictions.ipynb](https://github.com/yoon-thiri04/Smart-doc-ai/blob/master/notebooks/test_predictions.ipynb) | Inference experiments and an LLM example | 5 |
| notebooks/tfidf_vectorizer.joblib | Fitted text-to-number converter | 6 |
| models/logistic_regression.pkl | Saved Logistic Regression model | 6 |
| models/svm.pkl | Saved SVM model | 6 |
| models/random_forest.pkl | Saved Random Forest model | 6 |
| data/sample.txt, sample_doc.docx, sample_pdf.pdf | Manual test files | 7 |
| reports/classification_results.csv | Example notebook output | 8 |

There are no automated tests. The notebooks are the main executable experiments.

## 4. Read app.py in this order

Source: https://github.com/yoon-thiri04/Smart-doc-ai/blob/master/app.py

### 4.1 Page setup and Streamlit execution

The imports load Streamlit, document readers, pandas, joblib, LangChain, FPDF, PyPDF2, and other libraries. The page configuration sets the browser title and wide layout.

Streamlit reruns the Python script from top to bottom when a widget changes. In this app, widgets act as the control flow: uploading a file creates a result, selecting a model changes the selected prediction, and pressing the analysis button calls the LLM.

### 4.2 clean_text

Lines 25-29 lowercase text and remove characters outside ASCII letters, digits, and whitespace.

This is preprocessing. Training and inference must use the same preprocessing. Otherwise, the model would see different feature meanings at training time and prediction time.

The current function also removes non-ASCII characters. Burmese, accented characters, and many other writing systems are therefore discarded.

### 4.3 extract_text

Lines 32-54 convert three file formats into one plain string:

- TXT: read bytes and decode UTF-8;
- DOCX: join all paragraph text using python-docx;
- PDF: use PyPDF2 to join text extracted from every page.

The machine-learning code only receives a string, so it does not need to know which file format produced it.

This is text extraction, not OCR. A scanned PDF may contain only images and no text layer, so PyPDF2 may return empty text.

### 4.4 load_assets

Lines 57-68 use joblib.load to restore the vectorizer and three trained models. Streamlit cache_resource reuses these objects across reruns.

The paths are relative to the working directory. Start the app from the repository root:

~~~bash
streamlit run app.py
~~~

Otherwise paths such as models/svm.pkl may not be found.

### 4.5 generate_llm_analysis

Lines 71-95 create a LangChain ChatOpenAI client pointed at Together AI's OpenAI-compatible endpoint.

The settings are:

- model: meta-llama/Llama-3-8b-chat-hf;
- key: st.secrets["TOGETHER_API_KEY"];
- endpoint: https://api.together.xyz/v1;
- temperature: chosen by the user.

The prompt includes the first 2,000 characters of the document and the selected category. It asks for a title, a title explanation, and a category explanation.

The parameter is misspelled as tempeature. Python accepts it, but it should be renamed to temperature.

### 4.6 create_pdf_report

Lines 97-126 create an in-memory PDF with FPDF. The function writes each model prediction and confidence, then adds the LLM analysis if available. The resulting PDF bytes are passed to Streamlit's download button.

The PDF is a report, not a copy of the uploaded document.

### 4.7 main

The main function:

1. shows the title and upload widget;
2. waits until a file is uploaded;
3. extracts raw text and cleans it;
4. shows up to 500 characters as a preview;
5. loads the saved assets;
6. transforms the cleaned text with the fitted vectorizer;
7. calls predict and predict_proba on each classifier;
8. displays a pandas table;
9. lets the user select a model;
10. calls the LLM when the button is pressed;
11. treats the first line of the response as the title;
12. offers the PDF report.

The original file stays in memory. The app does not save the upload to disk.

## 5. Machine-learning concepts

### 5.1 Supervised classification

Each training example has an input and an answer:

~~~text
input:  article text
label:  business, sport, politics, tech, or entertainment
~~~

The model learns patterns from labelled examples. During inference it sees new text and estimates its label.

### 5.2 Train/test split

The notebook uses an 80/20 split with a fixed random seed:

~~~python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
~~~

X is the text and y is the category. The training split fits the models. The test split estimates how well they work on unseen examples.

The notebook does not use stratify=y, so an improvement is to preserve class proportions explicitly.

### 5.3 TF-IDF

Models need numbers, not paragraphs. TfidfVectorizer converts each document into a sparse vector whose columns represent words.

TF-IDF combines:

- term frequency: how often a word occurs in this document;
- inverse document frequency: how rare the word is across all documents.

Very common words receive less weight. Words that distinguish one kind of article from another can receive more weight.

The notebook creates:

~~~python
TfidfVectorizer(stop_words="english", max_features=5000)
~~~

The fitted vocabulary and IDF values are saved as notebooks/tfidf_vectorizer.joblib.

During inference the app must call transform, not fit_transform. Fitting on an uploaded document would create a new vocabulary and produce columns that no longer match the trained models.

### 5.4 The three classifiers

| Model | Concept | Role in this project |
|---|---|---|
| Logistic Regression | Learns weighted evidence for each class and produces class probabilities | Fast text baseline |
| SVM with probability enabled | Learns separating boundaries between classes | Strong comparison model for high-dimensional text |
| Random Forest | Combines many decision trees | Nonlinear comparison model |

The app compares the models but does not combine them into an ensemble. The user chooses which model's label is sent to the LLM.

### 5.5 Probability is not certainty

The UI takes the largest value from model.predict_proba. That value is an estimate, not proof that the label is correct. A document can be confidently misclassified, especially if it is unlike the training data.

The SVM probability setting adds probability estimation during training. Those values still should not be treated as perfectly calibrated probabilities.

## 6. AI and document concepts

### File parsing versus OCR

PyPDF2 reads a PDF's existing text layer. It does not recognize letters in a scanned image. Supporting scanned PDFs requires OCR, followed by the same cleaning and classification steps.

### Prompt construction

PromptTemplate fills the text and category placeholders before the request is sent to the LLM. The category comes from the selected classifier. The document text comes from the upload.

Uploaded text can contain instructions that look like prompts. A production app should treat LLM output as untrusted and consider prompt injection, privacy, redaction, length limits, and logging.

### Temperature

Lower temperature usually gives more repeatable output. Higher temperature usually gives more varied output. Temperature affects the LLM response, not the scikit-learn prediction.

## 7. Run it locally

### Prerequisites

- Python 3.9 or newer;
- a virtual environment;
- internet access for package installation and Together AI;
- a Together AI API key for the LLM feature.

### Install

~~~bash
git clone https://github.com/yoon-thiri04/Smart-doc-ai.git
cd Smart-doc-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

On Windows PowerShell:

~~~powershell
.venv/Scripts/Activate.ps1
~~~

### Configure Together AI

Create .streamlit/secrets.toml:

~~~toml
TOGETHER_API_KEY = "replace-with-your-key"
~~~

Do not commit the key. The repository ignores .streamlit/ and .env.

### Start

~~~bash
streamlit run app.py
~~~

Upload data/sample.txt first. Then try data/sample_doc.docx and data/sample_pdf.pdf. You should see the preview, three model rows, a model selector, a temperature slider, and the analysis button.

The PDF download appears only after the LLM request succeeds.

## 8. Re-run model training

The training notebook uses paths such as ../data and ../models, so run Jupyter with notebooks/ as the working directory:

~~~bash
cd notebooks
jupyter notebook train_models.ipynb
~~~

Run the cells in this order:

1. Load ../data/bbc-text.csv.
2. Keep category and text, then rename category to label.
3. Clean the text.
4. Split text and labels into training and test sets.
5. Fit TF-IDF on X_train only.
6. Transform X_train and X_test.
7. Fit Logistic Regression, SVM, and Random Forest.
8. Save the models into ../models/.
9. Save the vectorizer into the current notebooks/ directory.
10. Run accuracy, classification reports, and confusion matrices.

The vectorizer and all classifiers are a matched artifact set. If you change preprocessing or vectorizer settings, retrain and replace all of them together.

To test saved inference:

~~~bash
cd notebooks
jupyter notebook test_predictions.ipynb
~~~

That notebook writes reports/classification_results.csv and includes another Together AI example.

## 9. Function reference

| Function | Input | Output | Main dependency |
|---|---|---|---|
| clean_text(text) | Python string | Cleaned string | re |
| extract_text(uploaded_file) | Uploaded file with name and read | Text string or None | PyPDF2, python-docx |
| load_assets() | None | Vectorizer plus model dictionary | joblib, Streamlit cache |
| generate_llm_analysis(text, category, temperature) | Text, label, numeric temperature | LLM response string | LangChain, Together AI |
| create_pdf_report(predictions, llm_analysis) | Prediction list and optional analysis | PDF bytes | FPDF |
| main() | Streamlit state | Renders the app | Streamlit |

A prediction item currently looks like:

~~~python
{
    "Model": "SVM",
    "Prediction": "business",
    "Confidence": "88.30%",
}
~~~

A better internal contract would store confidence as a float and format it only for display.

## 10. Limitations and bugs worth studying

1. Formatted confidence strings are used for ranking. A value such as "88.30%" should be compared numerically, not as text.
2. The best-prediction message adds a second percent sign because the stored value already contains one.
3. If extraction fails and returns None, clean_text(raw_text) can fail on .lower().
4. File extension checks are case-sensitive. A PDF named with an uppercase suffix may be rejected.
5. A PDF page may return None from extract_text, which can make joining page text fail.
6. The first line of the LLM response is assumed to be a title, but the prompt does not enforce a strict response format.
7. requirements.txt does not pin versions. Future LangChain or OpenAI client versions may change behavior.
8. There are no automated tests. Manual Streamlit upload is the main test path.
9. joblib and pickle-based model files must come from a trusted source because loading them can execute code.
10. Uploaded text is sent to an external LLM provider. A production version needs a privacy policy and clear handling of confidential documents.
11. README claims and actual labels differ: the implementation uses the five BBC labels listed at the start.

## 11. Refactoring exercises

### Exercise A: make inference testable

Move the core prediction logic out of the UI:

~~~python
def classify_text(text, assets):
    cleaned = clean_text(text)
    vector = assets["vectorizer"].transform([cleaned])
    return [
        {
            "model": name,
            "prediction": model.predict(vector)[0],
            "confidence": float(model.predict_proba(vector)[0].max()),
        }
        for name, model in assets["models"].items()
    ]
~~~

Then write unit tests without starting Streamlit.

### Exercise B: fix numeric ranking

Use numeric confidence internally:

~~~python
best_result = max(result, key=lambda item: item["confidence"])
confidence_text = f"{best_result['confidence']:.2%}"
~~~

Format only when rendering the UI or PDF.

### Exercise C: improve extraction

Use Path(name).suffix.lower(), guard missing PDF text, reject empty documents, and show a useful error.

### Exercise D: improve evaluation

Use stratify=y, record accuracy and macro F1 for each model, and save the vectorizer settings with the evaluation results.

### Exercise E: make the LLM output structured

Ask for fields such as title, title_reason, and category_reason. Validate the result before displaying it or putting it into the PDF.

## 12. Recommended study plan

1. Python: functions, dictionaries, list comprehensions, exceptions, imports, and file-like objects. Start with clean_text, extract_text, and create_pdf_report.
2. Streamlit: reruns, widgets, caching, uploaders, expanders, and download buttons. Then read main.
3. Pandas: DataFrames and how prediction dictionaries become the results table.
4. Text classification: train/test split, labels, TF-IDF, sparse matrices, fit, transform, predict, and predict_proba.
5. Model evaluation: accuracy, precision, recall, macro F1, and confusion matrices. Do not select a model from one confidence score alone.
6. Serialization: why the vectorizer and classifiers must be loaded as a compatible set.
7. LLM integration: prompts, API-compatible endpoints, temperature, length limits, privacy, and structured output.
8. Production engineering: pinned dependencies, tests, error handling, logging, and deployment.

## 13. Checkpoint questions

- Why does inference call transform instead of fit_transform?
- What would happen if a new vectorizer were fitted on every upload?
- Why must training and inference share the same clean_text logic?
- What are the five actual labels?
- Why can a scanned PDF fail while a text PDF works?
- What does probability enabled change for the SVM?
- Why is probability not proof of correctness?
- Which prediction is sent to the LLM?
- What data leaves the local machine when analysis is generated?
- Why should "88.30%" not be used for numeric ranking?

If you can answer these and complete Exercises A-C, you understand the core of this codebase.

## 14. Glossary

- Artifact: a saved training output, such as a vectorizer or classifier file.
- Classification: choosing one label from a fixed set.
- Confidence or probability: the model's estimate for a class, not a guarantee.
- Inference: using a trained model on new input.
- LLM: large language model used for title and explanation generation.
- Preprocessing: converting raw input into the form expected by a model.
- Sparse matrix: a matrix where most entries are zero.
- TF-IDF: a word-weighting method based on frequency and rarity.
- Vectorizer: the fitted object that converts text into the same numeric columns used during training.

## 15. Best next step

Run the app with data/sample.txt. Then open notebooks/train_models.ipynb and trace one article from the CSV through cleaning, TF-IDF, model prediction, and the final Streamlit table. After that, implement Exercise A so the prediction path is independent of the UI.
