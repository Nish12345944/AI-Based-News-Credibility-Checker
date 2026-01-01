from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from textblob import TextBlob
import re
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# Trusted news sources
TRUSTED_SOURCES = {
    'reuters', 'bbc', 'cnn', 'nytimes', 'washingtonpost', 'guardian', 
    'ap', 'npr', 'pbs', 'abc', 'cbs', 'nbc', 'wsj', 'bloomberg',
    'associated press', 'new york times', 'washington post', 'the guardian'
}

# Suspicious/clickbait words
SUSPICIOUS_WORDS = {
    'shocking', 'unbelievable', 'amazing', 'incredible', 'you won\'t believe',
    'doctors hate', 'this one trick', 'secret', 'exposed', 'revealed',
    'breaking', 'urgent', 'must see', 'viral', 'gone wrong'
}

class NewsCredibilityAnalyzer:
    def __init__(self):
        self.model = self._create_model()
    
    def _create_model(self):
        # Create a simple Random Forest model
        # In a real application, this would be trained on actual data
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Generate some sample training data for demonstration
        X_sample = np.random.rand(1000, 10)
        y_sample = np.random.randint(0, 2, 1000)
        model.fit(X_sample, y_sample)
        
        return model
    
    def extract_features(self, title, content, source=''):
        """Extract features from the news article"""
        full_text = f"{title} {content}"
        
        # Basic text statistics
        word_count = len(full_text.split())
        char_count = len(full_text)
        title_length = len(title)
        
        # Punctuation analysis
        exclamation_count = full_text.count('!')
        question_count = full_text.count('?')
        caps_ratio = sum(1 for c in full_text if c.isupper()) / len(full_text) if full_text else 0
        
        # Sentiment analysis
        blob = TextBlob(full_text)
        sentiment_polarity = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity
        
        # Suspicious words count
        suspicious_word_count = sum(1 for word in SUSPICIOUS_WORDS 
                                  if word.lower() in full_text.lower())
        
        # Source credibility
        trusted_source = 1 if source.lower() in TRUSTED_SOURCES else 0
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'title_length': title_length,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'caps_ratio': caps_ratio,
            'sentiment_polarity': sentiment_polarity,
            'sentiment_subjectivity': sentiment_subjectivity,
            'suspicious_word_count': suspicious_word_count,
            'trusted_source': trusted_source
        }
    
    def analyze_credibility(self, title, content, source=''):
        """Analyze the credibility of a news article"""
        features = self.extract_features(title, content, source)
        
        # Convert features to array for model prediction
        feature_array = np.array([list(features.values())]).reshape(1, -1)
        
        # Get prediction probability
        credibility_prob = self.model.predict_proba(feature_array)[0][1]
        
        # Apply rule-based adjustments
        credibility_score = self._apply_rules(features, credibility_prob)
        
        # Generate explanation
        explanation = self._generate_explanation(features, credibility_score)
        
        # Determine credibility level
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
            'features': features
        }
    
    def _apply_rules(self, features, base_score):
        """Apply rule-based adjustments to the ML prediction"""
        score = base_score
        
        # Boost for trusted sources
        if features['trusted_source']:
            score = min(1.0, score + 0.2)
        
        # Penalize excessive suspicious words
        if features['suspicious_word_count'] > 3:
            score = max(0.0, score - 0.3)
        
        # Penalize excessive caps
        if features['caps_ratio'] > 0.3:
            score = max(0.0, score - 0.2)
        
        # Penalize excessive punctuation
        if features['exclamation_count'] > 5:
            score = max(0.0, score - 0.1)
        
        # Boost for balanced sentiment
        if abs(features['sentiment_polarity']) < 0.3:
            score = min(1.0, score + 0.1)
        
        return score
    
    def _generate_explanation(self, features, score):
        """Generate human-readable explanation"""
        explanation = []
        
        if features['trusted_source']:
            explanation.append("✓ Source appears to be from a trusted news outlet")
        else:
            explanation.append("⚠ Source is not from a recognized trusted outlet")
        
        if features['suspicious_word_count'] == 0:
            explanation.append("✓ No obvious clickbait language detected")
        else:
            explanation.append(f"⚠ Contains {features['suspicious_word_count']} suspicious/clickbait words")
        
        if features['caps_ratio'] < 0.1:
            explanation.append("✓ Appropriate use of capitalization")
        else:
            explanation.append("⚠ Excessive use of capital letters detected")
        
        if features['exclamation_count'] <= 2:
            explanation.append("✓ Moderate use of exclamation marks")
        else:
            explanation.append("⚠ Excessive use of exclamation marks")
        
        if abs(features['sentiment_polarity']) < 0.5:
            explanation.append("✓ Balanced emotional tone")
        else:
            explanation.append("⚠ Highly emotional or biased tone detected")
        
        if score >= 0.8:
            explanation.append("✓ Overall assessment: High credibility")
        elif score >= 0.6:
            explanation.append("⚠ Overall assessment: Moderate credibility")
        else:
            explanation.append("❌ Overall assessment: Low credibility - verify through other sources")
        
        return explanation

# Initialize analyzer
analyzer = NewsCredibilityAnalyzer()

@app.route('/app')
def serve_app():
    with open('index.html', 'r') as f:
        return f.read()

@app.route('/')
def home():
    return jsonify({
        'message': 'AI News Credibility Checker API',
        'version': '1.0',
        'endpoints': {
            'POST /analyze': 'Analyze news article credibility',
            'GET /health': 'Check API health'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'API is running'})

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
        
        # Analyze credibility
        result = analyzer.analyze_credibility(title, content, source)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)