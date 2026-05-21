from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_KEY = "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"
client = Groq(api_key=GROQ_KEY)

class ListingRequest(BaseModel):
    title: str
    tags: str
    description: str

class ChatRequest(BaseModel):
    message: str
    analysis: dict

def analyze_listing(title, tags, description, model_name):
    prompt = f"""You are an expert Etsy SEO consultant with years of experience helping sellers rank higher in Etsy search.

Analyze this Etsy listing in detail:

Title: {title}
Tags: {tags}  
Description: {description}

Respond ONLY with this exact JSON, no extra text:
{{
    "title_score": 7,
    "tags_score": 6,
    "description_score": 8,
    "overall_score": 7,
    "title_feedback": "detailed specific feedback",
    "tags_feedback": "detailed specific feedback",
    "description_feedback": "detailed specific feedback",
    "top_suggestions": ["specific actionable suggestion 1", "specific actionable suggestion 2", "specific actionable suggestion 3"],
    "rewritten_title": "your improved version of the title",
    "rewritten_tags": "your improved comma separated tags"
}}"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        raw = response.choices[0].message.content.strip()
        clean = raw.replace('```json', '').replace('```', '').strip()
        start = clean.find('{')
        end = clean.rfind('}') + 1
        return json.loads(clean[start:end])
    except Exception as e:
        return None

@app.post("/analyze")
async def analyze(request: ListingRequest):
    llama_large = analyze_listing(request.title, request.tags, request.description, "llama-3.3-70b-versatile")
    llama_small = analyze_listing(request.title, request.tags, request.description, "llama-3.1-8b-instant")

    def avg(key):
        scores = []
        if llama_large and key in llama_large:
            scores.append(llama_large[key])
        if llama_small and key in llama_small:
            scores.append(llama_small[key])
        return round(sum(scores) / len(scores), 1) if scores else 0

    return {
        "llama_large": llama_large,
        "llama_small": llama_small,
        "consensus": {
            "title_score": avg("title_score"),
            "tags_score": avg("tags_score"),
            "description_score": avg("description_score"),
            "overall_score": avg("overall_score")
        }
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    prompt = f"""You are an Etsy SEO expert. The user just got this analysis of their listing:

{json.dumps(request.analysis, indent=2)}

The user is asking: {request.message}

Answer helpfully and specifically based on their listing analysis. Be conversational and practical."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"response": "Sorry I couldn't process that. Try again!"}

@app.get("/")
async def root():
    return {"status": "Etsy SEO API running!"}