from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from training import prediction

class Text(BaseModel): 
    text: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Success"}

@app.post("/detect_lang")
async def detect_lang(text: Text) -> JSONResponse:
    raw_text = text.text
    language = prediction(raw_text)
    return JSONResponse({"Language": language})

if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8000)