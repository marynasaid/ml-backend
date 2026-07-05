from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq
import os
import json

router = APIRouter()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================
# REQUEST MODEL (OLD STYLE)
# =========================
class AIRequest(BaseModel):
    cycle_data: dict


# =========================
# LOG HELPER
# =========================
def log(title, data):
    print("\n" + "=" * 60)
    print(f"📌 {title}")
    print("=" * 60)
    print(json.dumps(data, indent=2, ensure_ascii=False))


# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are a cycle intelligence engine.

Analyze cycle_data and return ONLY a short insight.
No JSON, no markdown, no explanations.
"""


# =========================
# MAIN ENDPOINT (RESTORED)
# =========================
@router.post("/ai_daily_insight")
def ai_daily_insight(data: AIRequest):

    try:
        log("INCOMING REQUEST", data.model_dump())

        cycle_data = data.cycle_data

        log("CYCLE DATA", cycle_data)

        # =========================
        # PROMPT
        # =========================
        prompt = f"""
Cycle data:
{json.dumps(cycle_data, indent=2, ensure_ascii=False)}

Task:
Give a short personalized insight.
"""

        # =========================
        # GROQ REQUEST
        # =========================
        request_payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3
        }

        log("GROQ REQUEST", request_payload)

        response = client.chat.completions.create(**request_payload)

        ai_text = response.choices[0].message.content

        log("AI RESPONSE", {"insight": ai_text})

        return {
            "insight": ai_text
        }

    except Exception as e:
        log("ERROR", {"error": str(e)})

        return {
            "error": "AI failed",
            "detail": str(e)
        }