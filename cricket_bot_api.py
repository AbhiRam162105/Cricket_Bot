from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
from cricket_bot_data import WikipediaDocumentProcessor, CricketAssistant


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


class UserInput(BaseModel):
    text: str


@app.get("/api/v1/health")
async def health():
    return {"response": "Alive and well my friend !"}


@app.post("/chat")
async def chat_endpoint(user_input: UserInput):
    assistant = CricketAssistant(vector_store)
    response = assistant.handle_user_input(user_input.text)
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
