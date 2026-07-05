from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq
import os
import json

router = APIRouter()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================
# REQUEST MODEL
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
# MAIN ENDPOINT
# =========================
@router.post("/ai_daily_insight")
def ai_daily_insight(data: AIRequest):

    try:
        log("INCOMING REQUEST", data.model_dump())

        cycle_data = data.cycle_data

        log("CYCLE DATA", cycle_data)

        # =========================
        # SAFE PROMPT BUILD (NO f-string CURVY BUGS)
        # =========================
        cycle_data_str = json.dumps(cycle_data, indent=2, ensure_ascii=False)

        prompt = (
            "Cycle data:\n"
            f"{cycle_data_str}\n\n"
            "Task:\n"
            "You are a women's health insight engine. Your job is to generate a short, highly personalized, calm, non-diagnostic insight based on cycle model outputs.\n\n"
            "Do NOT provide medical advice or diagnosis. Do NOT assume pregnancy or disease. Focus only on probabilistic interpretation and self-awareness.\n\n"
            "Inputs:\n"
            "- current_cycle_day: integer (day of current menstrual cycle)\n"
            "- cycle_state: dictionary of probabilities\n"
            "- stability: float (0–1)\n"
            "- next_period: integer (predicted days until next period)\n\n"
            "Rules:\n"
            "1. Identify the most likely cycle phase = argmax(cycle_state).\n"
            "2. Use secondary probabilities only if close (within 0.15).\n"
            "3. Interpret stability:\n"
            "   - >0.75 stable\n"
            "   - 0.4–0.75 moderate\n"
            "   - <0.4 irregular (use cautious language)\n"
            "4. Include:\n"
            "   - current phase\n"
            "   - one body/mental signal\n"
            "   - one timing insight\n"
            "5. Tone: supportive, factual, non-alarming\n"
            "6. Max 90 words\n\n"
            "Output format:\n"
            "Insight:\n"
            "Body signal:\n"
            "Cycle outlook:\n"
            "Stability note:\n"
        )

        # =========================
        # GROQ REQUEST
        # =========================
        request_payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        log("GROQ REQUEST", request_payload)

        response = client.chat.completions.create(**request_payload)

        ai_text = response.choices[0].message.content

        log("AI RESPONSE", {"insight": ai_text})

        return {"insight": ai_text}

    except Exception as e:
        log("ERROR", {"error": str(e)})

        return {
            "error": "AI failed",
            "detail": str(e)
        }