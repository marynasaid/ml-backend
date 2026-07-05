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
        # =========================
        # 🔥 RAW REQUEST DEBUG (ВАЖНЕЙШЕЕ)
        # =========================
        print("\n🔥 AI ENDPOINT HIT")
        print("\n📦 FULL REQUEST BODY:")
        print(data.model_dump())

        cycle = data.cycle_data

        print("\n🔥 FLUTTER RAW INPUT (cycle_data):")
        print(json.dumps(cycle, indent=2, ensure_ascii=False))

        # =========================
        # SNAPSHOTS ONLY
        # =========================
        snapshots = cycle.get("snapshots", [])

        print("\n📊 SNAPSHOTS TYPE CHECK:")
        print(type(snapshots), "COUNT:", len(snapshots))

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
        # GROQ REQUEST
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

        print("\n🚀 GROQ REQUEST:")
        print(json.dumps(request_payload, indent=2, ensure_ascii=False))

        # =========================
        # CALL GROQ
        # =========================
        response = client.chat.completions.create(**request_payload)

        ai_text = response.choices[0].message.content

        # =========================
        # RESPONSE DEBUG
        # =========================
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