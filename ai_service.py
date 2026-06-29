from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
import os

router = APIRouter()


# 🔥 DEBUG: проверка ключа на Render
print("OPENAI KEY LOADED:", os.getenv("OPENAI_API_KEY") is not None)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AIRequest(BaseModel):
    cycle_data: dict


@router.post("/ai_daily_insight")
def ai_daily_insight(data: AIRequest):

    try:
        cycle = data.cycle_data

        prompt = f"""
You are a menstrual cycle assistant.

User cycle data:
{cycle}

Give:
- short daily insight
- energy level
- recommendation
- tone: friendly, not medical
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful cycle health assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return {
            "insight": response.choices[0].message.content
        }

    except Exception as e:
        print("AI ERROR:", str(e))
        return {
            "error": "AI failed",
            "detail": str(e)
        }