from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import StringIO

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSV URL
CSV_URL = "https://raw.githubusercontent.com/Mehranhamza/ai-model/main/all_hadiths_clean.csv"

# Load and Preprocess Data
def load_data():
    response = requests.get(CSV_URL)
    response.raise_for_status()  # Raise error if request fails
    data = pd.read_csv(StringIO(response.text))
    return data

# Preprocess Text
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

# Initialize Dataset
data = load_data()
data["clean_text"] = data["text_en"].apply(preprocess_text)
data = data.dropna(subset=["clean_text"]).reset_index(drop=True)

# Feature Extraction
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data["clean_text"])

# Request Model
class Query(BaseModel):
    query: str

# Utility Function
def get_similar_hadees(query, tfidf_matrix, tfidf_vectorizer, top_n=50):
    query_vector = tfidf_vectorizer.transform([preprocess_text(query)])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    similar_hadees_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    similar_hadees = data.iloc[similar_hadees_indices][["hadith_no", "text_en", "source"]]
    return similar_hadees

# Routes
@app.post("/get_similar_hadees")
async def simple_post(query: Query):
    try:
        if not query.query:
            raise HTTPException(status_code=400, detail="Query is required.")

        similar_hadees = get_similar_hadees(query.query, tfidf_matrix, tfidf_vectorizer)
        result = similar_hadees.to_dict(orient="records")
        return {"similar_hadees": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/getresult")
async def get_result():
    return {"similar_hadees": "Good"}
