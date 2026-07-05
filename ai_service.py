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
You are a women's health insight engine. Your job is to generate a short, highly personalized, calm, non-diagnostic insight based on cycle model outputs.

Do NOT provide medical advice or diagnosis. Do NOT assume pregnancy or disease. Focus only on probabilistic interpretation and self-awareness.

Inputs:
- current_cycle_day: integer (day of current menstrual cycle)
- cycle_state: dictionary of probabilities (sum may be 0–1 normalized or unnormalized)
    {
      "menstrual": float,
      "follicular": float,
      "ovulation": float,
      "luteal": float
    }
- stability: float (0–1), where:
    1 = very stable cycle pattern over time
    0 = highly irregular cycle patterns
- next_period: integer (predicted days until next period)

Rules:
1. Identify the most likely cycle phase = argmax(cycle_state).
2. Use secondary probabilities only if they are close (within 0.15 of max).
3. Interpret stability:
   - >0.75 = stable pattern
   - 0.4–0.75 = moderately variable
   - <0.4 = irregular pattern (use cautious language)
4. Always include:
   - current inferred phase
   - one physical/mental wellbeing pattern typical for that phase (neutral tone)
   - one personalized timing insight based on cycle_day and next_period
5. Tone:
   - supportive, factual, non-alarming
   - no exaggeration, no wellness clichés
   - no moral framing ("you should", "you must")

Output format:
Return EXACTLY this structure:

Insight:
[1–2 sentences summary of current phase inference]

Body signal:
[1 sentence about typical physiological or emotional patterns aligned with probabilities]

Cycle outlook:
[1 sentence using current_cycle_day + next_period to describe temporal position in cycle]

Stability note:
[1 sentence interpreting stability level and what it implies about predictability]

Do not add extra sections.
Keep total length under 90 words.
  
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