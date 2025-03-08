import os
import json
import uuid
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import markdown
import re

# Configuration
GEMINI_API_KEY = "AIzaSyDGpNmvskXEAeOH6hG_BtT8GR043tMREYk"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Flask app
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# RAG Model Initialization
print("🚀 Initializing RAG System...")

# Load medical guidelines dataset
print("📂 Loading dataset...")
dataset = load_dataset("epfl-llm/guidelines", split="train")
TITLE_COL = "title"
CONTENT_COL = "clean_text"

# Initialize models
print("🤖 Loading AI models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline(
    "question-answering", model="distilbert-base-cased-distilled-squad"
)

# Build FAISS index
print("🔍 Building FAISS index...")


def embed_text(batch):
    combined_texts = [
        f"{title} {content[:200]}"
        for title, content in zip(batch[TITLE_COL], batch[CONTENT_COL])
    ]
    return {"embeddings": embedder.encode(combined_texts, show_progress_bar=False)}


dataset = dataset.map(embed_text, batched=True, batch_size=32)
dataset.add_faiss_index(column="embeddings")


# Processing Functions
def format_response(text):
    """Convert Markdown text to HTML for proper frontend display."""
    return markdown.markdown(text)


def extract_patient_info(report):
    """Extract patient information using QA pipeline."""
    questions = [
        "What is the patient's name?",
        "What is the patient's age?",
        "What is the patient's gender?",
        "What are the current symptoms?",
        "What is the medical history?",
    ]

    answers = {}
    for q in questions:
        result = qa_pipeline(question=q, context=report)
        if q == "What is the patient's name?":
            answers["name"] = result["answer"] if result["score"] > 0.1 else "Unknown"
        elif q == "What is the patient's age?":
            answers["age"] = result["answer"] if result["score"] > 0.1 else "Unknown"
        elif q == "What is the patient's gender?":
            answers["gender"] = result["answer"] if result["score"] > 0.1 else "Unknown"
        elif q == "What are the current symptoms?":
            answers["symptoms"] = (
                result["answer"] if result["score"] > 0.1 else "Not specified"
            )
        elif q == "What is the medical history?":
            answers["history"] = (
                result["answer"] if result["score"] > 0.1 else "Not specified"
            )

    return answers


def summarize_report(report):
    """Generate a clinical summary using QA and Gemini model."""
    patient_info = extract_patient_info(report)

    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Create clinical summary from:
    - Name: {patient_info['name']}
    - Age: {patient_info['age']}
    - Gender: {patient_info['gender']}
    - Symptoms: {patient_info['symptoms']}
    - History: {patient_info['history']}
    
    Format as: "[{patient_info['name']}] is a [{patient_info['age']}]-year-old [{patient_info['gender']}] with [{patient_info['history']}], presenting with [{patient_info['symptoms']}]"
    Add relevant medical context."""

    summary_text = model.generate_content(prompt).text.strip()

    # Return both the formatted summary and the extracted patient info
    return format_response(summary_text), patient_info


def get_reference_url(source_name):
    """Convert source names to actual URLs."""
    # Define mappings for common medical guidelines sources
    url_mappings = {
        "CDC": "https://www.cdc.gov/guidelines/",
        "WHO": "https://www.who.int/publications/guidelines/",
        "NIH": "https://www.nih.gov/health-information/guidelines/",
        "ADA": "https://diabetes.org/clinical-guidance/",
        "AHA": "https://professional.heart.org/guidelines/",
        "AAFP": "https://www.aafp.org/clinical-recommendations/",
        "ACR": "https://www.rheumatology.org/practice-quality/clinical-support/clinical-practice-guidelines/",
        "NICE": "https://www.nice.org.uk/guidance/",
        "ACP": "https://www.acponline.org/clinical-information/guidelines/",
        "JAMA": "https://jamanetwork.com/journals/jama/clinical-guidelines/",
        "NEJM": "https://www.nejm.org/medical-guidelines",
        "Lancet": "https://www.thelancet.com/clinical/diseases",
        "BMJ": "https://www.bmj.com/uk/clinicalguidelines",
        "Mayo Clinic": "https://www.mayoclinic.org/medical-professionals/clinical-updates/",
    }

    # Check if the source matches any known sources
    for key, url in url_mappings.items():
        if key.lower() in source_name.lower():
            # If it's a known provider, use the mapped URL
            return {"name": source_name, "url": url}

    # For unmatched sources, try to extract URLs if they exist
    url_pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
    urls = re.findall(url_pattern, source_name)
    if urls:
        return {"name": source_name, "url": urls[0]}

    # For anything else, create a Google search URL for the source
    search_url = f"https://www.google.com/search?q={source_name.replace(' ', '+')}+medical+guidelines"
    return {"name": source_name, "url": search_url}


def rag_retrieval(query, k=3):
    """Retrieve relevant guidelines using FAISS."""
    query_embedding = embedder.encode([query])
    scores, examples = dataset.get_nearest_examples("embeddings", query_embedding, k=k)
    return [
        {
            "title": title,
            "content": content[:1000],
            "source": examples.get("source", ["N/A"] * len(examples[TITLE_COL]))[i],
            "score": float(score),
        }
        for i, (title, content, score) in enumerate(
            zip(
                examples[TITLE_COL],
                examples[CONTENT_COL],
                scores,
            )
        )
    ]


def generate_recommendations(report):
    """Generate treatment recommendations with RAG context."""
    guidelines = rag_retrieval(report)
    context = "Relevant Clinical Guidelines:\n" + "\n".join(
        [f"• {g['title']}: {g['content']} [Source: {g['source']}]" for g in guidelines]
    )

    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Generate treatment recommendations using these guidelines:
    {context}
    
    Patient Presentation:
    {report}
    
    Format with:
    - Bold section headers
    - Clear bullet points
    - Evidence markers [Guideline #]
    - Risk-benefit analysis
    - Include references to the sources provided where applicable
    """
    recommendations = model.generate_content(prompt).text.strip()

    # Extract references with URLs
    reference_objects = []
    for g in guidelines:
        if g["source"] != "N/A":
            reference_objects.append(get_reference_url(g["source"]))

    return format_response(recommendations), reference_objects


def generate_risk_assessment(summary):
    """Generate risk assessment using the summary."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Analyze clinical risk:
    {summary}
    
    Output format:
    Risk Score: 0-100
    Alert Level: 🔴 High/🟡 Medium/🟢 Low
    Key Risk Factors: bullet points
    Recommended Actions: bullet points"""
    return format_response(model.generate_content(prompt).text.strip())


# Flask Endpoints
@app.route("/upload-txt", methods=["POST"])
def handle_upload():
    """Handle text file upload and return processed data."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file or not file.filename.endswith(".txt"):
        return jsonify({"error": "Invalid file, must be a .txt file"}), 400

    try:
        content = file.read().decode("utf-8")
        if not content.strip():
            return jsonify({"error": "File is empty"}), 400

        summary, patient_info = summarize_report(content)
        recommendations, references = generate_recommendations(content)
        risk_assessment = generate_risk_assessment(summary)

        return jsonify(
            {
                "session_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "patient_info": patient_info,  # Send extracted patient info directly
                "recommendations": recommendations,
                "risk_assessment": risk_assessment,
                "references": references,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


# Serve static files
@app.route("/")
def serve_index():
    """Serve the index.html file."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve other static files from the frontend directory."""
    return send_from_directory(app.static_folder, path)


# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
