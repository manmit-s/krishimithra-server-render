# main.py - KrishiMithra FastAPI Server for Render Deployment
import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image
import io
import uvicorn
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="KrishiMithra API",
    description="AI-powered farming assistant using Google Gemini",
    version="1.0.0"
)

# Configure CORS for all origins (required for Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Flutter app domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Configure Google Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Farming advisor system prompt
FARMING_SYSTEM_PROMPT = """You are KrishiMithra, a Digital Krishi Advisor specifically designed to help farmers in India and around the world.

IMPORTANT GUIDELINES:
- You ONLY answer questions related to farming, agriculture, crops, irrigation, soil management, fertilizers, pesticides, plant diseases, weather impacts, livestock, and agricultural practices.
- If asked about anything outside agriculture/farming, politely redirect: "I'm KrishiMithra, your farming advisor. Please ask me about crops, agriculture, or farming-related topics."
- Always provide practical, actionable advice that farmers can implement.
- Consider Indian farming conditions, climate, and traditional practices when relevant.
- Be helpful, encouraging, and supportive to farmers.
- Use simple, clear language that farmers can easily understand.
- For image analysis, focus on identifying crops, diseases, pests, soil conditions, or farming equipment.
- Provide solutions that are cost-effective and accessible to small-scale farmers.

Remember: You are here to help farmers grow better crops and improve their agricultural practices."""

# Pydantic models
class TextPrompt(BaseModel):
    prompt: str

class APIResponse(BaseModel):
    success: bool
    response: str
    error: Optional[str] = None

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "KrishiMithra API is running successfully!",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "text_chat": "/generate (POST)",
            "image_analysis": "/analyze-image (POST)",
            "documentation": "/docs"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": "gemini-1.5-flash",
        "service": "KrishiMithra API"
    }

# Text-based farming advice endpoint
@app.post("/generate", response_model=APIResponse)
async def generate_farming_advice(body: TextPrompt):
    """
    Generate farming advice from text prompt
    """
    try:
        if not body.prompt.strip():
            return APIResponse(
                success=False,
                response="",
                error="Please provide a farming-related question"
            )
        
        # Combine system prompt with user question
        full_prompt = f"{FARMING_SYSTEM_PROMPT}\n\nFarmer's Question: {body.prompt.strip()}"
        
        # Generate response using Gemini
        response = model.generate_content(full_prompt)
        
        if not response.text:
            return APIResponse(
                success=False,
                response="",
                error="No response generated. Please try again."
            )
        
        return APIResponse(
            success=True,
            response=response.text,
            error=None
        )
        
    except Exception as e:
        print(f"Error in generate_farming_advice: {str(e)}")
        return APIResponse(
            success=False,
            response="",
            error="Sorry, I'm having trouble processing your request. Please try again later."
        )

# Image analysis endpoint
@app.post("/analyze-image", response_model=APIResponse)
async def analyze_farming_image(
    file: UploadFile = File(...),
    prompt: str = Form("Analyze this farming-related image and provide agricultural advice")
):
    """
    Analyze farming/crop images and provide advice
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return APIResponse(
                success=False,
                response="",
                error="Please upload a valid image file"
            )
        
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Combine system prompt with user prompt for image analysis
        full_prompt = f"{FARMING_SYSTEM_PROMPT}\n\nFarmer's Question: {prompt}\n\nPlease analyze the attached image and provide specific farming advice based on what you observe."
        
        # Generate response with image analysis
        response = model.generate_content([full_prompt, image])
        
        if not response.text:
            return APIResponse(
                success=False,
                response="",
                error="Could not analyze the image. Please try again."
            )
        
        return APIResponse(
            success=True,
            response=response.text,
            error=None
        )
        
    except Exception as e:
        print(f"Error in analyze_farming_image: {str(e)}")
        return APIResponse(
            success=False,
            response="",
            error="Sorry, I couldn't process your image. Please try again with a clear farming-related image."
        )

# Ping endpoint for monitoring
@app.get("/ping")
async def ping():
    return {"ping": "pong", "timestamp": "ok"}

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
