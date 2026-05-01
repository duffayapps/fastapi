from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Gym AI Workout Plan Generator")

class Exercise(BaseModel):
    name: str
    sets: int
    reps: str
    weight: str
    rest_seconds: int


class WorkoutDay(BaseModel):
    day: str
    focus: str
    warmup: list[str]
    exercises: list[Exercise]
    cooldown: list[str]


class WorkoutPlan(BaseModel):
    weekly_schedule: list[WorkoutDay]


class WorkoutRequest(BaseModel):
    age: int
    goal: str
    experience_level: str
    days_per_week: int
    session_length_minutes: int
    equipment: list[str]
    injuries_or_limitations: list[str] = Field(default_factory=list)


@app.get("/")
def home():
    return {"message": "Gym AI Workout Plan API is running"}


@app.post("/generate-workout-plan", response_model=WorkoutPlan)
def generate_workout_plan(request: WorkoutRequest):
    prompt = f"""
You are a qualified fitness assistant.

Create a safe, intermediate-friendly gym workout plan.

STRICT RULES:
- Output must be SHORT and concise
- NO paragraphs
- NO explanations
- Use bullet-style structure
- Each exercise must be in this format:

"name": "Dumbbell Bench Press",
"sets": 3,
"reps": "10",
"weight": "15kg",
"rest_seconds": 60

- Keep all text minimal
- No coaching tips
- No long notes

User:
Age: {request.age}
Goal: {request.goal}
Experience: {request.experience_level}
Days: {request.days_per_week}
Session: {request.session_length_minutes} mins
Equipment: {request.equipment}
Injuries: {request.injuries_or_limitations}
"""

    try:
        response = client.responses.parse(
            model="gpt-4.1-mini",
            input=prompt,
            text_format=WorkoutPlan,
            max_output_tokens=1500
        )

        return response.output_parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
