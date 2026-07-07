from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq
import os
import json
import re

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
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


# =========================
# FACT BUILDER — весь анализ здесь, не в LLM
# =========================
def build_facts(cd: dict) -> dict:
    cycle = cd.get("cycle", {})
    stab = cd.get("stability", {})
    symptoms = cd.get("today_symptoms", [])
    history = cd.get("her_history", {})
    dq = cd.get("data_quality", {})

    score = stab.get("score") or 0.0
    if score > 0.75:
        stability_label = "stable"
    elif score >= 0.4:
        stability_label = "moderate"
    else:
        stability_label = "irregular"

    # заметные симптомы: value > 0.55 ИЛИ тренд "higher", топ-3 по силе
    notable = sorted(
        [
            s for s in symptoms
            if isinstance(s.get("value"), (int, float))
            and (
                s["value"] > 0.55
                or s.get("trend_vs_her_avg_this_phase") == "higher"
            )
        ],
        key=lambda s: -s["value"],
    )[:3]

    return {
        "current_day": cycle.get("current_day"),
        "phase_today": cycle.get("phase_today"),
        "next_period_in_days": cycle.get("next_period_in_days"),
        "stability_label": stability_label,
        "on_schedule": stab.get("on_schedule"),
        "deviation_days": stab.get("deviation_days"),
        "notable_symptoms": [
            {
                "name": s["key"].replace("_", " "),
                "level": round(s["value"], 2),
                "vs_her_usual": s.get("trend_vs_her_avg_this_phase"),
            }
            for s in notable
        ],
        "recurring_patterns": history.get("recurring_patterns", [])[:3],
        "cycles_logged": history.get("cycles_logged"),
        "days_logged_this_cycle": dq.get("days_logged_this_cycle"),
        "enough_history": bool(dq.get("enough_history")),
    }


# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """You are the insight writer for Luna, a menstrual cycle and mood tracking app.
You receive pre-computed FACTS about one user's current cycle state.
Write a short personal overview for her main insight card.

STRUCTURE: plain text, 2 short paragraphs, 50-80 words total.
- Paragraph 1 (1-2 sentences): where she is in her cycle (day, phase) and
  whether her cycle is running on schedule, using the stability facts.
- Paragraph 2: comment on today's notable symptoms. For each, use
  vs_her_usual: "higher" means stronger than her own average for this
  phase, "typical" means in line with her usual, "lower" means milder,
  "no_baseline" means there is no history to compare against yet — in that
  case describe the symptom without comparison. If recurring_patterns is
  non-empty, weave ONE of them in naturally ("as in your past cycles...").
  If notable_symptoms is empty, say today looks calm symptom-wise.

HARD RULES:
- Use ONLY the facts given. Never invent numbers, statistics, percentages,
  or population comparisons ("most women", "70% of women" — forbidden).
- Never compare her to other women. Compare only to her own history, or to
  general phase physiology ("common in the luteal phase" is allowed).
- Never diagnose or name conditions (no PMS, PMDD, PCOS, endometriosis etc.).
  Never recommend treatments, supplements, or medication. The only medical
  action you may suggest is "worth mentioning to your doctor" — and only if
  a symptom is "higher" AND appears in recurring_patterns.
- If enough_history is false, do not claim patterns; note the picture will
  get clearer as she logs more cycles.
- If stability_label is "irregular", use cautious wording about timing
  predictions.
- Tone: calm, warm, factual. Like a knowledgeable friend, not a doctor and
  not a cheerleader. No exclamation marks, no emoji, no headers, no labels
  like "Insight:" — just flowing plain text.
- Address her as "you". Never mention the app, JSON, data, facts, or AI.
- The FIRST sentence must work as a standalone headline.
"""

# страховка от диагнозов и популяционных процентов в ответе
FORBIDDEN = re.compile(
    r"(PMDD|PMS|PCOS|endometriosis|diagnos|\d+\s*%\s*of\s*women|most women)",
    re.IGNORECASE,
)


# =========================
# FALLBACK — шаблонный текст из фактов, если модель нарушила правила
# =========================
def fallback_text(f: dict) -> str:
    day = f.get("current_day")
    phase = f.get("phase_today") or "current"
    nxt = f.get("next_period_in_days")

    parts = []
    if day:
        parts.append(f"You're on day {day} of your cycle, in the {phase} phase.")
    else:
        parts.append(f"You're in the {phase} phase of your cycle.")

    if nxt is not None:
        parts.append(f"Your next period is expected in about {nxt} days.")

    if f.get("stability_label") == "stable":
        parts.append("Your recent cycles have been consistent.")
    elif f.get("stability_label") == "irregular":
        parts.append(
            "Your recent cycles have varied, so timing predictions are approximate."
        )

    if f.get("notable_symptoms"):
        names = ", ".join(s["name"] for s in f["notable_symptoms"])
        parts.append(f"Today you logged noticeable {names}.")

    return " ".join(parts)


# =========================
# MAIN ENDPOINT
# =========================
@router.post("/ai_daily_insight")
def ai_daily_insight(data: AIRequest):

    try:
        log("INCOMING REQUEST", data.model_dump())

        facts = build_facts(data.cycle_data)

        log("FACTS", facts)

        user_msg = (
            "FACTS:\n"
            + json.dumps(facts, indent=2, ensure_ascii=False)
            + "\n\nWrite the insight now."
        )

        request_payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.35,
            "max_tokens": 200,
        }

        log("GROQ REQUEST", request_payload)

        response = client.chat.completions.create(**request_payload)

        ai_text = (response.choices[0].message.content or "").strip()

        # страховка: пустой ответ или запрещённые слова -> fallback из фактов
        if not ai_text or FORBIDDEN.search(ai_text):
            log("FILTERED, USING FALLBACK", {"raw": ai_text})
            ai_text = fallback_text(facts)

        log("AI RESPONSE", {"insight": ai_text})

        return {"insight": ai_text}

    except Exception as e:
        log("ERROR", {"error": str(e)})

        return {
            "error": "AI failed",
            "detail": str(e)
        }