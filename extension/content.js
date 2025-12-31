// Content script for real-time analysis
let analysisOverlay = null;

function createAnalysisOverlay(result) {
    if (analysisOverlay) {
        analysisOverlay.remove();
    }
    
    const score = Math.round(result.credibility_score * 100);
    const level = result.credibility_level.toLowerCase();
    
    analysisOverlay = document.createElement('div');
    analysisOverlay.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        width: 300px;
        background: white;
        border: 2px solid ${getColorForLevel(level)};
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 10000;
        font-family: Arial, sans-serif;
        font-size: 14px;
    `;
    
    analysisOverlay.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <strong>🔍 Credibility: ${score}%</strong>
            <button onclick="this.parentElement.parentElement.remove()" style="border: none; background: none; font-size: 18px; cursor: pointer;">×</button>
        </div>
        <div style="color: ${getColorForLevel(level)}; font-weight: bold; margin-bottom: 8px;">
            ${result.credibility_level} Credibility
        </div>
        <div style="font-size: 12px; color: #666;">
            Language: ${result.language} | Bias: ${result.bias_analysis.direction}
        </div>
        <div style="margin-top: 8px; font-size: 12px;">
            Top factor: ${result.feature_importance[0].feature}
        </div>
    `;
    
    document.body.appendChild(analysisOverlay);
    
    setTimeout(() => {
        if (analysisOverlay) {
            analysisOverlay.style.opacity = '0.8';
        }
    }, 5000);
}

function getColorForLevel(level) {
    const colors = {
        'high': '#28a745',
        'medium': '#ffc107', 
        'low': '#fd7e14',
        'very-low': '#dc3545'
    };
    return colors[level] || '#6c757d';
}

// Auto-analyze news sites
const newsSites = ['cnn.com', 'bbc.com', 'reuters.com', 'nytimes.com', 'foxnews.com'];
if (newsSites.some(site => window.location.hostname.includes(site))) {
    setTimeout(autoAnalyze, 2000);
}

async function autoAnalyze() {
    const title = document.title;
    const content = Array.from(document.querySelectorAll('p')).map(p => p.textContent).join(' ').slice(0, 1000);
    
    if (title && content.length > 100) {
        try {
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: title,
                    content: content,
                    source: window.location.hostname
                })
            });
            
            const result = await response.json();
            createAnalysisOverlay(result);
        } catch (error) {
            console.log('Credibility analysis failed:', error);
        }
    }
}