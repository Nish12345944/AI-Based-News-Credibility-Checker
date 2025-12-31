from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from textblob import TextBlob
import pickle
import os

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

app = Flask(__name__)
CORS(app)

class NewsCredibilityAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.is_trained = False
        
    def extract_features(self, title, content, source=""):
        text = f"{title} {content}"
        
        # Basic features
        features = {
            'word_count': len(text.split()),
            'char_count': len(text),
            'title_length': len(title),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        }
        
        # Sentiment analysis
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        # Suspicious patterns
        suspicious_words = ['breaking', 'shocking', 'unbelievable', 'must read', 'click here', 'you won\'t believe']
        features['suspicious_word_count'] = sum(1 for word in suspicious_words if word.lower() in text.lower())
        
        # Source credibility
        trusted_sources = ['reuters', 'bbc', 'ap', 'cnn', 'nytimes', 'washingtonpost', 'guardian']
        features['trusted_source'] = 1 if any(src in source.lower() for src in trusted_sources) else 0
        
        return features
    
    def create_training_data(self):
        data = [
            {"title": "Scientists Discover New Exoplanet", "content": "Researchers at NASA have confirmed the discovery of a new exoplanet using advanced telescopic technology. The planet, located 100 light-years away, shows potential for habitability.", "source": "reuters", "credible": 1},
            {"title": "SHOCKING: Aliens Found on Earth!", "content": "You won't believe what happened next! Click here to see the unbelievable truth about aliens living among us! This will change everything!", "source": "unknown", "credible": 0},
            {"title": "Economic Report Shows Steady Growth", "content": "The latest quarterly economic report indicates steady growth in the manufacturing sector, with unemployment rates declining by 0.2%.", "source": "bbc", "credible": 1},
            {"title": "BREAKING: Celebrity Secret Revealed", "content": "This shocking revelation will change everything you thought you knew! Must read now! You won't believe this incredible secret!", "source": "gossip", "credible": 0},
            {"title": "Climate Study Published in Nature", "content": "A comprehensive study on climate change effects has been published in the peer-reviewed journal Nature, showing significant temperature increases.", "source": "guardian", "credible": 1},
            {"title": "Miracle Cure Discovered!", "content": "Doctors hate this one simple trick! Unbelievable results that will shock you! Click here to learn the secret they don't want you to know!", "source": "unknown", "credible": 0},
            {"title": "New COVID Variant Detected", "content": "Health officials have identified a new variant of COVID-19 with increased transmissibility. Vaccination remains the primary defense.", "source": "cnn", "credible": 1},
            {"title": "URGENT: Government Conspiracy Exposed", "content": "Breaking news that they don't want you to see! This shocking truth will blow your mind! Share before it gets deleted!", "source": "conspiracy", "credible": 0},
        ]
        return pd.DataFrame(data)
    
    def train_model(self):
        df = self.create_training_data()
        
        # Extract features
        features_list = []
        for _, row in df.iterrows():
            features = self.extract_features(row['title'], row['content'], row['source'])
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        X = features_df.values
        y = df['credible'].values
        
        # Train model
        self.model.fit(X, y)
        self.feature_names = list(features_df.columns)
        self.is_trained = True
        
        # Save model
        os.makedirs('models', exist_ok=True)
        with open('models/credibility_model.pkl', 'wb') as f:
            pickle.dump({'model': self.model, 'feature_names': self.feature_names}, f)
    
    def load_model(self):
        try:
            with open('models/credibility_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_names = data['feature_names']
                self.is_trained = True
        except FileNotFoundError:
            self.train_model()
    
    def predict_credibility(self, title, content, source=""):
        if not self.is_trained:
            self.load_model()
        
        features = self.extract_features(title, content, source)
        feature_vector = [features.get(name, 0) for name in self.feature_names]
        
        credibility_prob = self.model.predict_proba([feature_vector])[0][1]
        
        return {
            'credibility_score': float(credibility_prob),
            'credibility_level': self.get_credibility_level(credibility_prob),
            'explanation': self.generate_explanation(features, credibility_prob),
            'features': features
        }
    
    def get_credibility_level(self, score):
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def generate_explanation(self, features, score):
        explanations = []
        
        if features['trusted_source']:
            explanations.append("✓ Source appears to be from a trusted news outlet")
        else:
            explanations.append("⚠ Source credibility could not be verified")
        
        if features['suspicious_word_count'] > 0:
            explanations.append(f"⚠ Contains {features['suspicious_word_count']} suspicious words/phrases")
        
        if features['caps_ratio'] > 0.1:
            explanations.append("⚠ High ratio of capital letters may indicate sensationalism")
        
        if features['exclamation_count'] > 2:
            explanations.append("⚠ Excessive use of exclamation marks")
        
        if abs(features['sentiment_polarity']) > 0.5:
            explanations.append("⚠ Highly emotional language detected")
        
        if len(explanations) == 1 and features['trusted_source']:
            explanations.append("✓ No obvious credibility issues detected")
        
        return explanations

# Initialize analyzer
analyzer = NewsCredibilityAnalyzer()

@app.route('/')
def home():
    return send_from_directory('..', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_news():
    try:
        data = request.get_json()
        title = data.get('title', '')
        content = data.get('content', '')
        source = data.get('source', '')
        
        if not title or not content:
            return jsonify({"error": "Title and content are required"}), 400
        
        result = analyzer.predict_credibility(title, content, source)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_trained": analyzer.is_trained})

if __name__ == '__main__':
    analyzer.load_model()
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)