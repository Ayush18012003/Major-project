"""
Utility functions for Depression Detection Model
Handles model loading, text preprocessing, and predictions
"""

import re
import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class DepressionDetectionModel:
    """
    Manages depression detection model loading and inference
    """
    
    def __init__(self, model_path=None, tokenizer_path=None):
        """
        Initialize the model with paths to model and tokenizer files
        
        Args:
            model_path (str): Path to keras model file (.keras)
            tokenizer_path (str): Path to tokenizer pickle file (.pkl)
        """
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_sequence_length = 300
        self.model_loaded = False
        
    def load_model(self):
        """Load keras model and tokenizer from disk"""
        try:
            if self.model_path is None or self.tokenizer_path is None:
                # Use default paths if not specified
                current_dir = Path(__file__).parent
                self.model_path = self.model_path or current_dir / "depression_detection_model.keras"
                self.tokenizer_path = self.tokenizer_path or current_dir / "tokenizer.pkl"
            
            # Load model
            self.model = tf.keras.models.load_model(str(self.model_path))
            
            # Load tokenizer
            with open(str(self.tokenizer_path), 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            self.model_loaded = True
            return True, "Model loaded successfully"
        
        except FileNotFoundError as e:
            error_msg = f"Model file not found: {str(e)}"
            return False, error_msg
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            return False, error_msg
    
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model_loaded and self.model is not None and self.tokenizer is not None


def comprehensive_clean_text(text):
    """
    Comprehensive text cleaning for depression detection:
    - Removes URLs and links
    - Removes emojis and special Unicode characters
    - Removes HTML entities
    - Converts to lowercase
    - Removes extra whitespace
    - Removes special characters while preserving word boundaries
    - Removes single character words
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Remove URLs and links (http, https, www, etc.)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 2. Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # 3. Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 4. Remove hashtags but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 5. Remove emojis and special Unicode characters
    text = re.sub(r'[\U0001F300-\U0001F9FF]+', '', text)  # Emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Non-ASCII characters
    
    # 6. HTML entities/Unicode common symbols
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'")
    
    # 7. Convert to lowercase
    text = text.lower()
    
    # 8. Remove punctuation but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 9. Remove multiple spaces and clean up
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 10. Remove single character words
    text = ' '.join([word for word in text.split() if len(word) > 1])
    
    return text


def predict_depression(model_obj, text):
    """
    Predict depression classification for input text
    
    Args:
        model_obj (DepressionDetectionModel): Loaded model object
        text (str): Input text for prediction
        
    Returns:
        dict: Prediction results with keys:
            - 'success': bool - whether prediction was successful
            - 'class': str - 'Depression' or 'No Depression'
            - 'probability': float - probability of depression (0-1)
            - 'confidence': float - confidence in prediction (0-1)
            - 'error': str - error message if prediction failed
    """
    
    result = {
        'success': False,
        'class': None,
        'probability': None,
        'confidence': None,
        'error': None
    }
    
    try:
        # Validate input
        if not text or len(text.strip()) == 0:
            result['error'] = "Input text cannot be empty"
            return result
        
        if len(text.strip()) < 3:
            result['error'] = "Input text too short (minimum 3 characters)"
            return result
        
        # Check if model is loaded
        if not model_obj.is_model_loaded():
            result['error'] = "Model not loaded. Please load the model first."
            return result
        
        # Step 1: Clean text
        cleaned_text = comprehensive_clean_text(text)
        
        if len(cleaned_text.split()) == 0:
            result['error'] = "Text after cleaning is empty. Try with more meaningful content."
            return result
        
        # Step 2: Tokenize
        sequences = model_obj.tokenizer.texts_to_sequences([cleaned_text])
        
        # Step 3: Pad sequences
        padded = pad_sequences(sequences, maxlen=model_obj.max_sequence_length, 
                              padding='post', truncating='post')
        
        # Step 4: Make prediction
        prediction_prob = model_obj.model.predict(padded, verbose=0)[0][0]
        
        # Step 5: Classify
        threshold = 0.5
        predicted_class = 1 if prediction_prob >= threshold else 0
        
        # Prepare results
        result['success'] = True
        result['class'] = 'Depression Detected' if predicted_class == 1 else 'No Depression'
        result['probability'] = float(prediction_prob)
        result['confidence'] = float(prediction_prob) if predicted_class == 1 else float(1 - prediction_prob)
        
        return result
    
    except Exception as e:
        result['error'] = f"Prediction error: {str(e)}"
        return result


def get_model_info():
    """
    Get model information
    
    Returns:
        dict: Model metadata
    """
    return {
        'model_name': 'Depression Detection BiLSTM',
        'accuracy': '90.20%',
        'precision': '87.33%',
        'recall': '94.05%',
        'f1_score': '90.56%',
        'auc_roc': '0.9694',
        'model_type': 'Bidirectional LSTM',
        'framework': 'TensorFlow/Keras'
    }
