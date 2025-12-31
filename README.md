# AI-Based News Credibility Checker

An intelligent system that analyzes news articles and provides credibility scores using Natural Language Processing and Machine Learning techniques.

## Features

- **Real-time Analysis**: Instant credibility scoring of news articles
- **NLP Processing**: Advanced text analysis using NLTK and TextBlob
- **Machine Learning**: Random Forest classifier for credibility prediction
- **Feature Extraction**: Multiple indicators including sentiment, source credibility, and suspicious patterns
- **Interactive UI**: Modern HTML/CSS/JavaScript interface
- **REST API**: Flask backend with comprehensive endpoints

## Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **NLTK** - Natural Language Processing
- **Scikit-learn** - Machine Learning
- **TextBlob** - Sentiment Analysis
- **Pandas/NumPy** - Data processing

### Frontend
- **HTML5/CSS3** - User interface
- **JavaScript** - Interactive functionality
- **Fetch API** - HTTP client

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Backend Setup

1. Navigate to the project directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r ../requirements.txt
```

3. Run the Flask server:
```bash
python app.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

Simply open `index.html` in your web browser. No additional setup required.

## API Endpoints

### POST /analyze
Analyze news article credibility

**Request Body:**
```json
{
  "title": "Article title",
  "content": "Article content",
  "source": "News source (optional)"
}
```

**Response:**
```json
{
  "credibility_score": 0.85,
  "credibility_level": "High",
  "explanation": [
    "✓ Source appears to be from a trusted news outlet",
    "✓ No obvious credibility issues detected"
  ],
  "features": {
    "word_count": 150,
    "sentiment_polarity": 0.1,
    "suspicious_word_count": 0,
    "trusted_source": 1
  }
}
```

### GET /health
Check API health status

### GET /
API information and status

## How It Works

### Feature Extraction
The system analyzes multiple aspects of news articles:

1. **Text Statistics**: Word count, character count, title length
2. **Sentiment Analysis**: Emotional tone and subjectivity using TextBlob
3. **Suspicious Patterns**: Clickbait indicators, excessive punctuation
4. **Source Credibility**: Known trusted news sources
5. **Language Patterns**: Caps ratio, exclamation usage

### Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Features**: 10 key indicators of credibility
- **Training**: Sample dataset with labeled credible/non-credible articles
- **Output**: Probability score (0-1) converted to percentage

### Credibility Levels
- **High (80-100%)**: Highly credible content
- **Medium (60-79%)**: Moderately credible
- **Low (40-59%)**: Low credibility
- **Very Low (0-39%)**: Potentially unreliable

## Usage Example

1. Open `index.html` in your web browser
2. Enter article title and content in the form
3. Optionally specify the news source
4. Click "Analyze Credibility"
5. Review the credibility score and detailed explanation
6. Examine extracted features for transparency

## Sample Test Cases

### High Credibility Article
```
Title: "Scientists Discover New Exoplanet"
Content: "Researchers at NASA have confirmed the discovery of a new exoplanet using advanced telescopic technology."
Source: "reuters"
```

### Low Credibility Article
```
Title: "SHOCKING: Aliens Found on Earth!"
Content: "You won't believe what happened next! Click here to see the unbelievable truth!"
Source: "unknown"
```

## Project Structure

```
news-credibility-checker/
├── backend/
│   ├── app.py              # Flask application
│   └── models/             # Trained ML models
├── data/
│   └── news_dataset.csv    # Training dataset
├── requirements.txt        # Python dependencies
├── index.html             # Frontend interface
└── README.md              # Documentation
```

## Model Features

The ML model analyzes these key features:

- **word_count**: Total words in the article
- **char_count**: Total characters
- **title_length**: Length of the title
- **exclamation_count**: Number of exclamation marks
- **question_count**: Number of question marks
- **caps_ratio**: Ratio of capital letters
- **sentiment_polarity**: Emotional sentiment (-1 to 1)
- **sentiment_subjectivity**: Subjectivity score (0 to 1)
- **suspicious_word_count**: Count of clickbait words
- **trusted_source**: Whether source is from trusted outlet

## Future Enhancements

- Integration with real news APIs
- Advanced deep learning models (BERT, GPT)
- Fact-checking database integration
- Multi-language support
- Browser extension
- Mobile application
- Real-time news monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Disclaimer

This tool is for educational and research purposes. Always verify important news through multiple reliable sources.