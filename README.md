# 🧠 Depression Detection AI

A machine learning application that detects potential signs of depression from text using a Bidirectional LSTM deep learning model. Built with Streamlit for easy deployment and user interaction.

## 📋 Features

- **Deep Learning Model**: Bidirectional LSTM neural network with 90.20% accuracy
- **Text Preprocessing**: Comprehensive text cleaning pipeline (URLs, emojis, HTML entities removal)
- **Real-time Analysis**: Instant predictions with confidence scores
- **User-Friendly Interface**: Clean, intuitive Streamlit web application
- **Production-Ready**: Fully documented, tested, and deployment-ready code
- **Model Performance**:
  - ✅ Accuracy: 90.20%
  - ✅ Precision: 87.33%
  - ✅ Recall: 94.05%
  - ✅ F1-Score: 90.56%
  - ✅ AUC-ROC: 0.9694

## ⚠️ Disclaimer

**This tool is for educational and awareness purposes only.** It should **NOT** be used as a substitute for professional mental health evaluation by qualified healthcare providers. If you or someone you know is struggling with depression, please reach out to a mental health professional.

## 📁 Project Structure

```
.
├── app.py                              # ✨ Main Streamlit application
├── utils.py                            # ✨ Utility functions and model class
├── requirements.txt                    # 📦 Python dependencies
├── README.md                           # 📚 This file
├── LICENSE                             # ⚖️ MIT License
├── .env.example                        # ⚙️ Environment variables template
├── .gitignore                          # 🚫 Git ignore configuration
├── depression_detection_model.keras    # 🤖 Trained model weights
├── tokenizer.pkl                       # 🔤 Text tokenizer
├── model_metadata.json                 # 📊 Model metadata
├── balanced_20k.csv                    # 📈 Training dataset
├── Depression Detection DL.ipynb       # 📓 Original Jupyter notebook
└── [Visualizations: *.png files]       # 📸 EDA visualizations
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download the repository**
```bash
git clone https://github.com/Ayush18012003/Major-project.git
cd Major-project
```

2. **Create a virtual environment (recommended)**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`


## � Usage

1. **Enter Text**: Paste or type text in the input area
2. **Click "Analyze Text"**: The model will process and analyze the input
3. **View Results**: See the prediction class, probability, and confidence level
4. **Optional Settings**:
   - Toggle "Show text cleaning example" to see how text is preprocessed
   - Toggle "Show confidence level" to see detailed confidence metrics

### Example Inputs

**Example 1 (Potential Depression):**
```
I don't know how to continue anymore. Everything feels empty and meaningless. 
I can't sleep properly and I've lost interest in things I once enjoyed. 
Nothing brings me joy anymore.
```

**Example 2 (No Depression):**
```
Had a great day at work today! Finished my project ahead of schedule. 
Going out with friends this weekend. Feeling motivated and energetic!
```

## �📦 Dependencies

Core packages used in this project:

- **streamlit** - Web application framework
- **tensorflow** - Deep learning framework
- **keras** - Neural network API
- **numpy** - Numerical computing
- **nltk** - Natural language processing
- **scikit-learn** - Machine learning utilities
- **pandas** - Data manipulation
- **matplotlib, seaborn** - Visualization

See `requirements.txt` for specific versions.

## 🛠️ Project Components

### 1. `utils.py`

Contains utility functions and the `DepressionDetectionModel` class:

```python
# Initialize model
model = DepressionDetectionModel()
model.load_model()

# Make prediction
result = predict_depression(model, "Your text here")
print(result)
# Output: {
#     'success': True,
#     'class': 'Depression Detected',
#     'probability': 0.85,
#     'confidence': 0.85,
#     'error': None
# }
```

**Key Functions:**
- `comprehensive_clean_text(text)` - Cleans and preprocesses text
- `predict_depression(model_obj, text)` - Makes predictions
- `get_model_info()` - Returns model metadata

### 2. `app.py`

Streamlit web application with:
- Model loading and error handling
- Interactive text input
- Real-time predictions
- Confidence visualization
- Text cleaning demo
- Responsive UI with custom styling
- Comprehensive sidebar with model information

## 📊 Model Details

### Architecture
- **Type**: Bidirectional LSTM with Attention
- **Embedding Dimension**: 128
- **Vocabulary Size**: 10,000
- **Maximum Sequence Length**: 300 tokens
- **LSTM Units**: [100, 50]
- **Dense Units**: [64, 32]
- **Dropout Rate**: 0.3
- **L2 Regularization**: 0.001
- **Total Parameters**: 1,572,177

### Training
- **Dataset**: 20,000 balanced samples
- **Train/Val/Test Split**: 64% / 16% / 20%
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Binary Crossentropy
- **Epochs**: 8 (with early stopping)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Text Preprocessing Pipeline
The model uses a comprehensive 11-step cleaning process:
1. Remove URLs and links
2. Remove email addresses
3. Remove mentions (@username)
4. Remove hashtags (keep text)
5. Remove emojis
6. Remove non-ASCII characters
7. Handle HTML entities
8. Convert to lowercase
9. Remove punctuation
10. Remove extra whitespace
11. Remove single-character words