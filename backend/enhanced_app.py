from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from textblob import TextBlob
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from langdetect import detect
import requests
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)

class EnhancedNewsAnalyzer:
    def __init__(self):
        self.model = self._create_enhanced_model()
        self.vader = SentimentIntensityAnalyzer()
        self.bias_keywords = self._load_bias_keywords()
        self.feature_names = [
            'word_count', 'sentiment_polarity', 'suspicious_words', 'trusted_source',
            'caps_ratio', 'exclamation_count', 'political_bias', 'emotional_manipulation',
            'readability_score', 'source_diversity'
        ]
    
    def _create_enhanced_model(self):
        # Enhanced model with feature importance
        model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        # Simulate training with enhanced features
        X_sample = np.random.rand(2000, 10)
        y_sample = np.random.randint(0, 2, 2000)
        model.fit(X_sample, y_sample)
        return model
    
    def _load_bias_keywords(self):
        return {
            'left_bias': ['liberal', 'progressive', 'democrat', 'leftist', 'socialist'],
            'right_bias': ['conservative', 'republican', 'rightist', 'traditional', 'patriot'],
            'emotional_manipulation': ['outrageous', 'devastating', 'shocking', 'terrifying', 'unbelievable']
        }
    
    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return 'en'
    
    def analyze_bias(self, text):
        text_lower = text.lower()
        left_count = sum(1 for word in self.bias_keywords['left_bias'] if word in text_lower)
        right_count = sum(1 for word in self.bias_keywords['right_bias'] if word in text_lower)
        emotional_count = sum(1 for word in self.bias_keywords['emotional_manipulation'] if word in text_lower)
        
        bias_score = (right_count - left_count) / max(len(text.split()), 1)
        emotional_score = emotional_count / max(len(text.split()), 1)
        
        return {
            'political_bias': bias_score,
            'emotional_manipulation': emotional_score,
            'bias_direction': 'right' if bias_score > 0.01 else 'left' if bias_score < -0.01 else 'neutral'
        }
    
    def extract_enhanced_features(self, title, content, source=''):
        full_text = f"{title} {content}"
        language = self.detect_language(full_text)
        
        # Basic features
        word_count = len(full_text.split())
        caps_ratio = sum(1 for c in full_text if c.isupper()) / len(full_text) if full_text else 0
        exclamation_count = full_text.count('!')
        
        # Sentiment analysis
        blob = TextBlob(full_text)
        sentiment_polarity = blob.sentiment.polarity
        
        # Bias analysis
        bias_analysis = self.analyze_bias(full_text)
        
        # Suspicious words
        suspicious_words = ['shocking', 'unbelievable', 'secret', 'exposed']
        suspicious_count = sum(1 for word in suspicious_words if word.lower() in full_text.lower())
        
        # Source credibility
        trusted_sources = ['reuters', 'bbc', 'cnn', 'nytimes', 'ap']
        trusted_source = 1 if source.lower() in trusted_sources else 0
        
        # Readability (simplified)
        sentences = full_text.count('.') + full_text.count('!') + full_text.count('?')
        readability_score = word_count / max(sentences, 1)
        
        return {
            'word_count': word_count,
            'sentiment_polarity': sentiment_polarity,
            'suspicious_words': suspicious_count,
            'trusted_source': trusted_source,
            'caps_ratio': caps_ratio,
            'exclamation_count': exclamation_count,
            'political_bias': bias_analysis['political_bias'],
            'emotional_manipulation': bias_analysis['emotional_manipulation'],
            'readability_score': readability_score,
            'source_diversity': 0.5,  # Placeholder
            'language': language,
            'bias_direction': bias_analysis['bias_direction']
        }
    
    def get_feature_importance(self, features):
        feature_array = np.array([list(features.values())[:10]]).reshape(1, -1)
        importances = self.model.feature_importances_
        
        feature_importance = []
        for i, (name, importance) in enumerate(zip(self.feature_names, importances)):
            feature_importance.append({
                'feature': name,
                'importance': float(importance),
                'value': list(features.values())[i] if i < len(features) else 0
            })
        
        return sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
    
    def analyze_credibility(self, title, content, source=''):
        features = self.extract_enhanced_features(title, content, source)
        
        # Model prediction
        feature_array = np.array([list(features.values())[:10]]).reshape(1, -1)
        credibility_prob = self.model.predict_proba(feature_array)[0][1]
        
        # Apply enhanced rules
        credibility_score = self._apply_enhanced_rules(features, credibility_prob)
        
        # Feature importance
        feature_importance = self.get_feature_importance(features)
        
        # Generate explanation
        explanation = self._generate_enhanced_explanation(features, credibility_score)
        
        # Determine level
        if credibility_score >= 0.8:
            level = "High"
        elif credibility_score >= 0.6:
            level = "Medium"
        elif credibility_score >= 0.4:
            level = "Low"
        else:
            level = "Very Low"
        
        return {
            'credibility_score': credibility_score,
            'credibility_level': level,
            'explanation': explanation,
            'features': features,
            'feature_importance': feature_importance,
            'language': features['language'],
            'bias_analysis': {
                'direction': features['bias_direction'],
                'political_bias_score': features['political_bias'],
                'emotional_manipulation_score': features['emotional_manipulation']
            }
        }
    
    def _apply_enhanced_rules(self, features, base_score):
        score = base_score
        
        # Enhanced rule-based adjustments
        if features['trusted_source']:
            score = min(1.0, score + 0.25)
        
        if features['suspicious_words'] > 2:
            score = max(0.0, score - 0.3)
        
        if features['emotional_manipulation'] > 0.05:
            score = max(0.0, score - 0.2)
        
        if abs(features['political_bias']) > 0.1:
            score = max(0.0, score - 0.15)
        
        if features['caps_ratio'] > 0.2:
            score = max(0.0, score - 0.1)
        
        return score
    
    def _generate_enhanced_explanation(self, features, score):
        explanation = []
        
        # Language detection
        lang_names = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German'}
        explanation.append(f"📝 Detected language: {lang_names.get(features['language'], features['language'])}")
        
        # Bias analysis
        if features['bias_direction'] != 'neutral':
            explanation.append(f"⚠ Political bias detected: {features['bias_direction']}-leaning")
        else:
            explanation.append("✓ No significant political bias detected")
        
        # Emotional manipulation
        if features['emotional_manipulation'] > 0.03:
            explanation.append("⚠ High emotional manipulation detected")
        else:
            explanation.append("✓ Low emotional manipulation")
        
        # Source credibility
        if features['trusted_source']:
            explanation.append("✓ Source from trusted outlet")
        else:
            explanation.append("⚠ Source not from recognized trusted outlet")
        
        # Suspicious content
        if features['suspicious_words'] > 0:
            explanation.append(f"⚠ Contains {features['suspicious_words']} suspicious words")
        else:
            explanation.append("✓ No suspicious language detected")
        
        return explanation

# Initialize analyzer
analyzer = EnhancedNewsAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_news():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        title = data.get('title', '').strip()
        content = data.get('content', '').strip()
        source = data.get('source', '').strip()
        
        if not title or not content:
            return jsonify({'error': 'Title and content are required'}), 400
        
        result = analyzer.analyze_credibility(title, content, source)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Enhanced API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)