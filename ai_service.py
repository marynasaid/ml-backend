from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq
import os
import json

router = APIRouter()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class AIRequest(BaseModel):
    cycle_data: dict


@router.post("/ai_daily_insight")
def ai_daily_insight(data: AIRequest):

    try:
        cycle = data.cycle_data

        # =========================
        # SNAPSHOTS ONLY
        # =========================
        snapshots = cycle.get("snapshots", [])

        latest = snapshots[-1] if snapshots else None
        history = snapshots[:-1] if len(snapshots) > 1 else []

        latest_text = (
            f"id={latest.get('id')} | phase={latest.get('phase_of_day')} | axes={latest.get('axes', {})}"
            if latest else "no data"
        )

        history_text = "\n".join([
            f"{s.get('id')} | phase={s.get('phase_of_day')} | axes={s.get('axes', {})}"
            for s in history[-20:]
        ])

        # =========================
        # PROMPT
        # =========================
        prompt = f"""
Ты — женская покровительница и мягкий аналитик состояния цикла.

LATEST:
{latest_text}

HISTORY:
{history_text}

ЗАДАЧА:
Дай короткий прогноз и анализ.
"""

        # =========================
        # FULL GROQ DEBUG (NEW)
        # =========================
        request_payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a warm cycle pattern analyst."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        print("\n🚀 ============== FULL GROQ REQUEST ==============")
        print(json.dumps(request_payload, indent=2, ensure_ascii=False))
        print("==================================================\n")

        # =========================
        # OLD DEBUG INPUT
        # =========================
        print("\n📦 ============== AI INPUT DEBUG ==============")
        print("\n📊 RAW SNAPSHOTS COUNT:", len(snapshots))
        print("\n📊 LATEST SNAPSHOT:", latest)

        print("\n📩 PROMPT SENT TO AI:")
        print(prompt)
        print("======================================\n")

        # =========================
        # GROQ CALL
        # =========================
        response = client.chat.completions.create(**request_payload)

        # =========================
        # RESPONSE DEBUG
        # =========================
        ai_text = response.choices[0].message.content

        print("\n🤖 AI RESPONSE:")
        print(ai_text)
        print("======================================\n")

        return {
            "insight": ai_text
        }

    except Exception as e:
        print("\n❌ ERROR:")
        print(str(e))

        return {
            "error": "AI failed",
            "detail": str(e)
        }