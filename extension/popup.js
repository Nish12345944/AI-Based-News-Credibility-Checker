async function analyzeCurrentPage() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: extractPageContent
    });
    
    const content = results[0].result;
    if (!content.title || !content.text) {
        document.getElementById('result').innerHTML = '<p>No article content found on this page.</p>';
        return;
    }
    
    try {
        const response = await fetch('http://localhost:5000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: content.title,
                content: content.text,
                source: content.source
            })
        });
        
        const result = await response.json();
        displayResult(result);
    } catch (error) {
        document.getElementById('result').innerHTML = '<p>Error: Backend not running</p>';
    }
}

function extractPageContent() {
    const title = document.title || document.querySelector('h1')?.textContent || '';
    const paragraphs = Array.from(document.querySelectorAll('p')).map(p => p.textContent).join(' ');
    const source = window.location.hostname;
    
    return {
        title: title.slice(0, 200),
        text: paragraphs.slice(0, 2000),
        source: source
    };
}

function displayResult(result) {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('result').style.display = 'block';
    
    const score = Math.round(result.credibility_score * 100);
    const level = result.credibility_level.toLowerCase().replace(' ', '-');
    
    document.getElementById('score').innerHTML = `${score}% ${result.credibility_level}`;
    document.getElementById('score').className = `score ${level}`;
    
    document.getElementById('biasInfo').innerHTML = `
        <strong>Language:</strong> ${result.language}<br>
        <strong>Bias:</strong> ${result.bias_analysis.direction}<br>
        <strong>Emotional Manipulation:</strong> ${(result.bias_analysis.emotional_manipulation_score * 100).toFixed(1)}%
    `;
    
    const topFeatures = result.feature_importance.slice(0, 3);
    document.getElementById('featureImportance').innerHTML = `
        <strong>Top Factors:</strong><br>
        ${topFeatures.map(f => `
            <div>${f.feature}: ${(f.importance * 100).toFixed(1)}%</div>
            <div class="feature-bar">
                <div class="feature-fill" style="width: ${f.importance * 100}%"></div>
            </div>
        `).join('')}
    `;
}

document.addEventListener('DOMContentLoaded', analyzeCurrentPage);