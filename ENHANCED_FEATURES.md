# Enhanced Features Installation Guide

## New Features Added

### 1. Bias Detection & Multi-language Support
- Political bias analysis (left/right/neutral)
- Emotional manipulation detection
- Language detection (English, Spanish, French, German)

### 2. Feature Importance Analysis
- Shows which factors most influenced the credibility score
- Visual importance bars for top 5 features
- Transparency in AI decision-making

### 3. Browser Extension
- Real-time analysis on news websites
- Popup interface with quick results
- Auto-analysis on major news sites

## Installation Steps

### Backend Setup
1. Install enhanced dependencies:
```bash
pip install -r requirements_enhanced.txt
```

2. Run the enhanced backend:
```bash
cd backend
python enhanced_app.py
```

### Browser Extension Setup
1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `extension` folder
5. The extension icon will appear in your toolbar

### Frontend Updates
The main interface now includes:
- Bias analysis section
- Feature importance visualization
- Multi-language support indicators

## Usage

### Web Interface
1. Open `index.html` in your browser
2. Enter article content
3. View enhanced results with bias analysis and feature importance

### Browser Extension
1. Visit any news website
2. Click the extension icon
3. Get instant credibility analysis with bias detection

## Enhanced Features Details

### Bias Detection
- **Political Bias**: Detects left/right political leaning
- **Emotional Manipulation**: Identifies manipulative language
- **Bias Direction**: Shows neutral/left/right classification

### Feature Importance
- **Visual Bars**: Shows relative importance of each factor
- **Top 5 Features**: Displays most influential factors
- **Transparency**: Explains AI decision-making process

### Multi-language Support
- **Auto-detection**: Automatically detects article language
- **Supported Languages**: English, Spanish, French, German
- **Language-aware Analysis**: Adjusts analysis based on language

## Testing the Features

### Test Political Bias Detection
```
Title: "Liberal Policies Destroying America"
Content: "Progressive leftist agenda is ruining our traditional values..."
```

### Test Emotional Manipulation
```
Title: "SHOCKING: You Won't Believe What Happened Next!"
Content: "This devastating news will terrify you. Unbelievable revelations..."
```

### Test Multi-language (Spanish)
```
Title: "Noticias importantes de España"
Content: "Las últimas noticias sobre política española..."
```

The system will now provide comprehensive analysis including credibility score, bias detection, feature importance, and language identification.