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
# CRISIS DETECTION (safety layer — runs BEFORE anything else)
# =========================
# If her notes contain signals of serious distress / self-harm ideation, we do
# NOT generate a breezy cycle summary. We return a gentle, non-clinical message
# that points her toward real human support. This is intentionally high-recall
# (better to over-trigger than miss). PMDD carries elevated suicidality risk in
# the luteal phase, so this matters.
CRISIS_PATTERNS = re.compile(
    r"\b("
    r"kill myself|killing myself|end my life|ending my life|end it all|"
    r"want to die|wanna die|better off dead|don'?t want to (be here|live)|"
    r"no reason to live|can'?t go on|can'?t do this anymore|"
    r"suicid|self[\s-]?harm|hurt myself|hurting myself|cut myself|cutting myself|"
    r"take my (own )?life|overdose|no way out|disappear forever"
    r")\b",
    re.IGNORECASE,
)

CRISIS_MESSAGE = (
    "Before anything about your cycle — some of what you wrote sounds really "
    "heavy, and you don't have to carry it alone. What you're feeling matters, "
    "and reaching out to someone you trust or a mental health professional can "
    "help. If you might be in danger right now, please contact your local "
    "emergency number or a crisis line in your country — for example, you can "
    "dial or text 988 in the US and Canada, or call 111 in the UK. You deserve "
    "support that goes beyond what an app can give."
)


def detect_crisis(cd: dict) -> bool:
    notes = []
    sa = cd.get("snapshot_analytics", {})
    notes += sa.get("notes_this_cycle", []) or []
    # also scan any legacy note fields, just in case
    for n in notes:
        if isinstance(n, str) and CRISIS_PATTERNS.search(n):
            return True
    return False


# =========================
# FACT BUILDER — all analysis here, not in the LLM
# =========================
def _severity_word(hard_days_this_cycle: int, avg_hard: float | None) -> str:
    """How this cycle compares to her own usual hard-day load."""
    if avg_hard is None:
        return "no_baseline"
    if hard_days_this_cycle == 0:
        return "calm"
    if hard_days_this_cycle > avg_hard * 1.5:
        return "harder_than_usual"
    if hard_days_this_cycle < avg_hard * 0.6:
        return "milder_than_usual"
    return "typical"


def build_facts(cd: dict) -> dict:
    cycle = cd.get("cycle", {})
    stab = cd.get("stability", {})
    symptoms = cd.get("today_symptoms", [])
    history = cd.get("her_history", {})
    dq = cd.get("data_quality", {})

    sa = cd.get("snapshot_analytics", {})
    stats = sa.get("cycle_stats", {})
    current = sa.get("current_cycle", {})
    recovery = sa.get("recovery_history", {})
    logging = sa.get("logging", {})

    # ── cycle timing ──
    mean_len = stats.get("mean_cycle_length")
    sd_len = stats.get("sd_cycle_length")
    has_cycle_stats = stats.get("has_cycle_stats", False)
    cycle_progress = stats.get("cycle_progress")
    deviation_days = stats.get("deviation_days")
    day_zscore = stats.get("day_zscore")
    schedule_status = stats.get("schedule_status", "unknown")
    is_flow = stats.get("is_flow", False)
    period_day = stats.get("period_day")
    mean_period = stats.get("mean_period_length")
    sd_period = stats.get("sd_period_length")

    # ── how hard is THIS cycle vs her usual, per section ──
    hard_by_section = current.get("hard_days_by_section", {}) or {}
    hard_by_axis = current.get("hard_days_by_axis", {}) or {}
    avg_hard_per_cycle = recovery.get("avg_hard_days_per_cycle", {}) or {}

    section_status = {}
    for sect in ("emotional", "body", "impact"):
        hd = hard_by_section.get(sect, 0)
        avg = avg_hard_per_cycle.get(sect)
        section_status[sect] = {
            "hard_days_this_cycle": hd,
            "usual_hard_days": round(avg, 1) if isinstance(avg, (int, float)) else None,
            "status": _severity_word(hd, avg if isinstance(avg, (int, float)) else None),
        }

    # sections that are currently hard, worst first
    hard_sections = sorted(
        [s for s in ("emotional", "body", "impact") if hard_by_section.get(s, 0) > 0],
        key=lambda s: -hard_by_section.get(s, 0),
    )

    # is she past the likely peak? (rough: hard days tend to ease near period)
    # we don't promise; we surface her typical run length instead.
    typical_run = (recovery.get("avg_hard_run_length") or {})

    # readable axis names for whichever axes are hard right now
    hard_axis_names = [k.replace("_", " ").replace("e ", "").replace("b ", "").replace("x ", "")
                       for k in hard_by_axis.keys()]

    # ── stability label from her own variation ──
    if has_cycle_stats and mean_len and sd_len is not None:
        # coefficient of variation → stable / moderate / irregular
        cov = (sd_len / mean_len) if mean_len else 1
        if cov < 0.09:
            stability_label = "stable"
        elif cov < 0.16:
            stability_label = "moderate"
        else:
            stability_label = "irregular"
    else:
        stability_label = "unknown"

    # ── notable symptoms today (value > 4 on 1..6, or trending higher) ──
    notable = sorted(
        [
            s for s in symptoms
            if isinstance(s.get("value"), (int, float))
            and (s["value"] > 4.0 or s.get("trend_vs_her_avg_this_phase") == "higher")
        ],
        key=lambda s: -s["value"],
    )[:3]

    return {
        # where she is
        "current_day": cycle.get("current_day"),
        "phase_today": cycle.get("phase_today"),
        "cycle_progress_pct": round(cycle_progress * 100) if isinstance(cycle_progress, (int, float)) else None,
        "next_period_in_days": cycle.get("next_period_in_days"),
        "is_flow": is_flow,
        "period_day": period_day,
        # timing norms
        "mean_cycle_length": mean_len,
        "sd_cycle_length": sd_len,
        "deviation_days": deviation_days,
        "day_zscore": day_zscore,
        "schedule_status": schedule_status,
        "mean_period_length": mean_period,
        "sd_period_length": sd_period,
        "stability_label": stability_label,
        # THIS cycle's difficulty vs her usual
        "section_status": section_status,
        "hard_sections_now": hard_sections,
        "hard_axis_names": hard_axis_names[:4],
        "typical_hard_run_days": typical_run,
        # notes as free context
        "notes_this_cycle": sa.get("notes_this_cycle", [])[:12],
        # today's notable symptoms
        "notable_symptoms": [
            {
                "name": s["key"].replace("_", " "),
                "level": round(s["value"], 1),
                "vs_her_usual": s.get("trend_vs_her_avg_this_phase"),
            }
            for s in notable
        ],
        # engagement / data sufficiency
        "cycles_logged": logging.get("cycles_logged", history.get("cycles_logged")),
        "days_logged_this_cycle": logging.get("days_logged_this_cycle", dq.get("days_logged_this_cycle")),
        "logging_rate_this_cycle": logging.get("logging_rate_this_cycle"),
        "enough_history": bool(logging.get("has_enough_history", dq.get("enough_history"))),
        "recurring_patterns": history.get("recurring_patterns", [])[:3],
    }


# =========================
# SYSTEM PROMPT — PMS / PMDD-aware support + brief cycle summary
# =========================
SYSTEM_PROMPT = """You are the insight writer for Luna, a menstrual-cycle and mood app used by
women who often experience PMS and PMDD. You receive pre-computed FACTS about
one user. Write a short, warm, personal note for her main insight card.

YOUR TWO JOBS, in this order:
1. A brief cycle summary: what day and phase she's in; whether her cycle is on
   schedule. CRITICAL — judge "on schedule" from schedule_status, NOT from the
   raw day count. schedule_status already accounts for HER OWN variation:
     * "on_track"     → within her normal range; reassure it's on time for her,
                        even if the raw day number looks high.
     * "slightly_off" → a little outside her usual range; mention gently.
     * "notably_off"  → well beyond her usual range (day_zscore > 2); note it
                        calmly, without alarm, and lean on stability_label for
                        how confident to sound.
   The same delay in days can be perfectly normal for one woman and unusual for
   another — schedule_status is what tells them apart. If is_flow is true,
   mention it's day {period_day} of her period and whether that fits her usual
   period length.
2. Emotional support based on her logged difficulty:
   - Read section_status (emotional / body / impact) and hard_sections_now.
   - If nothing is hard (all "calm"): reflect that things look steadier right
     now, gently and without over-celebrating.
   - If something IS hard: VALIDATE first (what she feels is real and hard),
     NORMALIZE (this is tied to her cycle phase and hormones, not a personal
     failing), then give PERSPECTIVE from HER OWN history.
     * status "typical": reassure that this matches her usual pattern and, per
       typical_hard_run_days, these stretches have passed before and will pass
       again — she has gotten through this.
     * status "harder_than_usual": acknowledge this cycle is heavier than her
       norm (e.g. more hard days than her usual). Do not minimize. Reassure it
       is still temporary and tied to this cycle.
     * status "no_baseline" (first cycle / no history): don't compare to a
       baseline. Simply reassure that difficult days in this phase are common
       and do pass, and that Luna will learn her personal pattern over time.

USING NOTES: notes_this_cycle are her own words. Use them ONLY as context to
sound specific and to guess what might be behind a change from her usual — you
may gently reflect a theme back ("you mentioned work has been heavy"). Never
quote them verbatim at length, never assume beyond what's written.

HARD RULES:
- Use ONLY the facts given. Never invent numbers, percentages, or population
  comparisons ("most women", "70% of women" are FORBIDDEN).
- Compare her only to HER OWN history, or to general phase physiology
  ("common in the luteal phase" is fine).
- Never diagnose or name conditions (no PMS, PMDD, PCOS, endometriosis, etc.).
  Never recommend treatments, supplements, or medication. The only medical
  suggestion allowed is "worth mentioning to your doctor" — and only if a
  section is "harder_than_usual" across the cycle, framed gently.
- Never promise a specific number of days until she feels better. You may say
  "in past cycles this kind of stretch has lasted around X days" using
  typical_hard_run_days, framed as her pattern, NOT a guarantee.
- If enough_history is false, do not claim patterns; say the picture gets
  clearer as she logs more. You may briefly, WARMLY note that logging helps
  Luna understand her — but never pressure, guilt, or push her to log more.
- If stability_label is "irregular" or "unknown", use cautious wording about
  timing.
- Tone: calm, warm, validating, factual — grounded mental-health support, like
  a knowledgeable friend. Not a doctor, not a cheerleader. No exclamation
  marks, no emoji, no headers, no labels. Address her as "you". Never mention
  the app internals, JSON, data, facts, or AI.

LENGTH: plain text, 2 short paragraphs, 60-100 words. The FIRST sentence must
work as a standalone headline.
"""

# safety net against diagnoses / population stats leaking into output
FORBIDDEN = re.compile(
    r"(PMDD|PMS|PCOS|endometriosis|diagnos|\d+\s*%\s*of\s*women|most women)",
    re.IGNORECASE,
)


# =========================
# FALLBACK — deterministic text from facts if the model breaks a rule
# =========================
def fallback_text(f: dict) -> str:
    day = f.get("current_day")
    phase = f.get("phase_today") or "current"
    parts = []

    if f.get("is_flow") and f.get("period_day"):
        parts.append(f"You're on day {f['period_day']} of your period, in the {phase} phase.")
    elif day:
        parts.append(f"You're on day {day} of your cycle, in the {phase} phase.")
    else:
        parts.append(f"You're in the {phase} phase of your cycle.")

    sched = f.get("schedule_status")
    if sched == "on_track":
        parts.append("This is within your normal range, so you're on time for you.")
    elif sched == "slightly_off":
        parts.append("This is running a little longer than your usual range.")
    elif sched == "notably_off":
        parts.append("This is beyond your usual range; timing may be less predictable this cycle.")

    hard = f.get("hard_sections_now") or []
    if not hard:
        parts.append("Things look steadier right now, symptom-wise.")
    else:
        sect = hard[0]
        status = (f.get("section_status", {}).get(sect, {}) or {}).get("status")
        if status == "harder_than_usual":
            parts.append(
                "This cycle has felt heavier than your usual — that's real, and "
                "it's tied to this phase, not to you. It will pass."
            )
        elif status == "no_baseline":
            parts.append(
                "Hard days in this phase are common and they do pass. As you keep "
                "logging, Luna will learn your own pattern."
            )
        else:
            parts.append(
                "What you're feeling is real and tied to your cycle phase. You've "
                "moved through stretches like this before, and you will again."
            )
    return " ".join(parts)


# =========================
# MAIN ENDPOINT
# =========================
@router.post("/ai_daily_insight")
def ai_daily_insight(data: AIRequest):
    try:
        log("INCOMING REQUEST", data.model_dump())

        # ── SAFETY: crisis check first ──
        if detect_crisis(data.cycle_data):
            log("CRISIS DETECTED → support message", {})
            return {"insight": CRISIS_MESSAGE, "crisis": True}

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
            "temperature": 0.4,
            "max_tokens": 260,
        }
        log("GROQ REQUEST", request_payload)

        response = client.chat.completions.create(**request_payload)
        ai_text = (response.choices[0].message.content or "").strip()

        if not ai_text or FORBIDDEN.search(ai_text):
            log("FILTERED, USING FALLBACK", {"raw": ai_text})
            ai_text = fallback_text(facts)

        log("AI RESPONSE", {"insight": ai_text})
        return {"insight": ai_text}

    except Exception as e:
        log("ERROR", {"error": str(e)})
        return {"error": "AI failed", "detail": str(e)}