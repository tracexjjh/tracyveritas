"""
Model 1 — Full ML Version for Railway.app
"""

from typing import List, Dict, Any
import re
from flask import Flask, request, jsonify

print("🚀 Starting Model 1 on Railway.app...")

# Import ML libraries - THEY WILL WORK on Railway!
import spacy
from transformers import pipeline
import torch

print("✅ All ML libraries imported successfully!")

app = Flask(__name__)

# Load models
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading T5 claim extractor...")
claim_extractor = pipeline(
    "text2text-generation",
    model="Babelscape/t5-base-summarization-claim-extractor",
    device=-1,  # Use CPU
    torch_dtype=torch.float32
)

print("✅ All models loaded successfully!")

# Entity mappings
WHO_LABELS = {"PERSON", "ORG", "NORP"}
WHERE_LABELS = {"GPE", "LOC", "FAC"}
WHEN_LABELS = {"DATE", "TIME"}
HOW_MUCH_LABELS = {"MONEY", "PERCENT", "QUANTITY", "CARDINAL"}

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    return [x for x in items if x and not (x in seen or seen.add(x))]

def extract_claims_from_text(
    article_id: str,
    text: str,
    max_claims: int = 4,
    min_score: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Full ML-powered claim extraction.
    """
    if not text.strip():
        return []

    try:
        # Generate claims with T5
        result = claim_extractor(
            text,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
        
        claim_text = result[0]['generated_text']
        raw_claims = [c.strip() for c in claim_text.split('.') if c.strip()]
        
        # Process each claim
        scored_claims = []
        for idx, claim in enumerate(raw_claims[:max_claims*2]):
            doc = nlp(claim)
            
            # Extract entities
            who, where, when, how_much = [], [], [], []
            for ent in doc.ents:
                if ent.label_ in WHO_LABELS:
                    who.append(ent.text)
                elif ent.label_ in WHERE_LABELS:
                    where.append(ent.text)
                elif ent.label_ in WHEN_LABELS:
                    when.append(ent.text)
                elif ent.label_ in HOW_MUCH_LABELS:
                    how_much.append(ent.text)
            
            # Simple scoring
            score = 0.3
            if who: score += 0.2
            if when: score += 0.2
            if where: score += 0.2
            if how_much: score += 0.1
            
            if score >= min_score:
                scored_claims.append({
                    "article_id": article_id,
                    "claim_id": f"{article_id}_{idx+1}",
                    "sent_ids": [idx],
                    "who": dedupe_preserve_order(who),
                    "what": [claim],
                    "when": dedupe_preserve_order(when),
                    "where": dedupe_preserve_order(where),
                    "how_much": dedupe_preserve_order(how_much),
                    "salience_score": round(score, 4),
                    "full_claim_text": claim
                })
        
        scored_claims.sort(key=lambda x: x["salience_score"], reverse=True)
        return scored_claims[:max_claims]
        
    except Exception as e:
        print(f"Error in ML processing: {e}")
        return []

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model": "full_ml_claim_extractor",
        "platform": "railway",
        "disk_space": "adequate_for_ml"
    })

@app.route('/extract-claims', methods=['POST'])
def api_extract_claims():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request body"}), 400
        
        article_id = data.get('article_id', 'default_article')
        text = data['text']
        max_claims = data.get('max_claims', 4)
        min_score = data.get('min_score', 0.5)
        
        claims = extract_claims_from_text(article_id, text, max_claims, min_score)
        
        return jsonify({
            "status": "success",
            "article_id": article_id,
            "claims_found": len(claims),
            "claims": claims
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return """
    <h1>🚀 Model 1 - Full ML Version</h1>
    <p>Running successfully on Railway.app with adequate disk space!</p>
    <p><strong>Endpoints:</strong></p>
    <ul>
        <li>GET <code>/health</code> - Health check</li>
        <li>POST <code>/extract-claims</code> - Extract claims from text</li>
    </ul>
    <p><strong>Example usage:</strong></p>
    <pre>
curl -X POST https://your-app.railway.app/extract-claims \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Your article text here...", "max_claims": 3}'
    </pre>
    """

if __name__ == '__main__':
    print("✅ Model 1 fully loaded and ready on Railway!")
    app.run(host='0.0.0.0', port=5000, debug=False)
