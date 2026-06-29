from fastapi import APIRouter

router = APIRouter()


@router.post("/insights")
async def insights(data: dict):

    is_flow = data.get("is_flow", 0)
    cycle_progress = data.get("cycle_progress", 0)

    if is_flow:
        text = "Menstrual phase: focus on rest and recovery."
    elif cycle_progress < 0.4:
        text = "Follicular phase: energy is rising."
    elif cycle_progress < 0.6:
        text = "Ovulation window: peak fertility period."
    else:
        text = "Luteal phase: body is stabilizing."

    return {"text": text}