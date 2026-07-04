from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq
import os

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

        if not snapshots:
            return {
                "insight": "Недостаточно данных для анализа."
            }

        # последний = текущий прогноз
        latest = snapshots[-1]

        # история
        history = snapshots[:-1]

        # формат истории (ограничиваем)
        history_text = "\n".join([
            f"{s.get('id')} | phase={s.get('phase_of_day')} | axes={s.get('axes', {})}"
            for s in history[-20:]
        ])

        latest_text = (
            f"id={latest.get('id')} | phase={latest.get('phase_of_day')} | axes={latest.get('axes', {})}"
        )

        prompt = f"""
Ты — женская покровительница и аналитик состояния цикла.

Ты не врач.
Ты даёшь мягкий, тёплый, но точный daily forecast.

---

ВАЖНАЯ ЛОГИКА ДАННЫХ:

- LATEST SNAPSHOT = текущее состояние и прогноз на сегодня
- SNAPSHOT HISTORY = прошлые прогнозы и динамика фаз
- AXES = симптомы (0..1), отражают физическое и эмоциональное состояние

AXES ПРИМЕР:
focus (headache)
social (cramps)
calm (fatigue)
calm1 (sleep_issue)
calm2 (mood_swings)
calm3 (food_cravings)
stress (stress)
bloating (bloating)

---

LATEST STATE:
{latest_text}

HISTORY:
{history_text}

---

ЗАДАЧА:

1. Daily note (1–2 предложения):
очень короткое состояние дня + ощущение

2. Phase interpretation:
что означает текущая фаза

3. Symptom dynamics:
- какие симптомы растут / падают
- особенно headache, stress, fatigue, mood swings
- найти 1–2 корреляции с фазами

4. Tomorrow hint:
очень короткий прогноз на завтра

---

СТИЛЬ:
- очень коротко
- как заметка в приложении
- тёплый тон "женской покровительницы"
- без медицины
- без длинных объяснений
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a warm, precise cycle pattern and symptom dynamics analyst. You write very short daily notes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            "insight": response.choices[0].message.content
        }

    except Exception as e:
        return {
            "error": "AI failed",
            "detail": str(e)
        }