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
        # FEATURE EXTRACTION LAYER
        # =========================
        cycle_progress = cycle.get("cycle_progress", 0)
        current_cycle_day = cycle.get("current_cycle_day", 0)
        last_cycle_length = cycle.get("last_cycle_length", 0)
        flow_length = cycle.get("flow_length", 0)
        avg_cycle_length = cycle.get("avg_cycle_length", 0)
        cycle_length_deviation = cycle.get("cycle_length_deviation", 0)

        prompt = f"""
You are a menstrual cycle data analyst.

You receive structured cycle metrics computed in the backend.

FEATURES (pre-extracted):
- current_cycle_day: {current_cycle_day}
- last_cycle_length: {last_cycle_length}
- cycle_progress: {cycle_progress}
- flow_length: {flow_length}
- avg_cycle_length: {avg_cycle_length}
- cycle_length_deviation: {cycle_length_deviation}

RAW DATA (JSON):
{cycle}

IMPORTANT FIELD DEFINITIONS:
- current_cycle_day: current day in cycle starting from 1
- last_cycle_length: previous full cycle length in days
- cycle_progress: current_cycle_day / last_cycle_length
  (can be > 1.0 if cycle is longer than average)
- flow_length: current period duration
- avg_cycle_length: average cycle length

TASK:
Analyze ONLY the provided data. Do NOT guess missing values.

Return:

1. Daily insight (1-2 sentences)
2. Cycle status interpretation (what stage user is in based on cycle_progress)
3. Energy level (low / medium / high)
4. Recommendation for today (practical, non-medical)
5. Pattern note (if cycle_progress or cycle_length_deviation looks unusual)

RULES:
- Be precise and data-driven
- No medical diagnosis
- No hallucinated data
- Use cycle_progress as main signal for timing
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise cycle data analytics assistant."
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