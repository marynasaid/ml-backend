from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq
import os
import json

router = APIRouter()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================
# REQUEST MODEL (NEW)
# =========================
class AIRequest(BaseModel):
    instructions: str   # overview | chart_1 | chart_2
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
# SYSTEM PROMPT (FIXED BEHAVIOR)
# =========================
SYSTEM_PROMPT = """
You are a cycle intelligence engine.

You MUST:
- analyze cycle_data
- follow instructions strictly
- return ONLY valid JSON
- do not add explanations or markdown
"""


# =========================
# MAIN ENDPOINT
# =========================
@router.post("/ai_daily_insight")
def ai_daily_insight(data: AIRequest):

    try:
        log("INCOMING REQUEST", data.model_dump())

        cycle_data = data.cycle_data
        instructions = data.instructions

        # =========================
        # BUILD CLEAN JSON INPUT
        # =========================
        ai_input = {
            "instructions": instructions,
            "cycle_data": cycle_data
        }

        log("AI INPUT (FINAL)", ai_input)

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
                    "content": json.dumps(ai_input, ensure_ascii=False)
                }
            ],
            "temperature": 0.3
        }

        log("GROQ PAYLOAD", request_payload)

        # =========================
        # CALL GROQ
        # =========================
        response = client.chat.completions.create(**request_payload)

        ai_text = response.choices[0].message.content

        # =========================
        # RAW OUTPUT LOG
        # =========================
        log("RAW AI RESPONSE", {"response": ai_text})

        # =========================
        # TRY PARSE JSON
        # =========================
        try:
            parsed = json.loads(ai_text)
        except:
            parsed = {
                "raw": ai_text,
                "parse_error": True
            }

        return {
            "instructions": instructions,
            "result": parsed
        }

    except Exception as e:
        log("ERROR", {"error": str(e)})

        return {
            "error": "AI failed",
            "detail": str(e)
        }