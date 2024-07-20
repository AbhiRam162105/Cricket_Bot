from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
from cricket_bot_data import WikipediaDocumentProcessor, CricketAssistant

# Define your CORS policy
# Replace "*" with your specific origin(s) in production
CORS_POLICY = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    print("Running initialization tasks before server starts.")
    processor = WikipediaDocumentProcessor(
        query="Cricket and everything related to cricket")
    processor.run()
    vector_store = processor.vectorstore
    print("Initialization complete.")
    yield
    print("Shutting Down.... Adios!!")

app = FastAPI(lifespan=lifespan)

# Add the CORSMiddleware to your FastAPI application
app.add_middleware(
    CORSMiddleware,
    **CORS_POLICY
)


class UserInput(BaseModel):
    text: str


@app.get("/api/v1/health")
async def health():
    return {"response": "Alive and well my friend !"}


@app.post("/chat")
async def chat_endpoint(user_input: UserInput):
    print(user_input)
    assistant = CricketAssistant(vector_store)
    response = assistant.handle_user_input(user_input.text)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
