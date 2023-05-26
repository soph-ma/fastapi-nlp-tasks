from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# import uvicorn
import nltk
from frequencies import create_freq_dist
from summarizer import Summarizer
# from keywords import KeyWordsExtractor
# from language_detector.training import prediction

# nltk.download('stopwords')
# nltk.download('punkt')

class Text(BaseModel): 
    text: str

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=origins,
    allow_headers=origins
    )

@app.get("/")
async def root():
    return {"message": "Success"}

@app.post("/freq_dict")
async def freq_dict(text: Text) -> JSONResponse:
    raw_text = text.text
    frequencies = create_freq_dist(raw_text)
    return JSONResponse({"Frequencies": frequencies})

@app.post("/summarize")
async def freq_dict(text: Text) -> JSONResponse:
    raw_text = text.text
    summarizer = Summarizer(raw_text)
    summary = summarizer.summarize()
    return JSONResponse({"Summary": summary})

# @app.post("/kwords")
# async def kwords(text: Text) -> JSONResponse: 
#     raw_text = text.text
#     kwords_extractor = KeyWordsExtractor(raw_text)
#     keywords = kwords_extractor.extract_keywords()
#     return JSONResponse({"Keywords": keywords})

# @app.post("/detect_lang")
# async def detect_lang(text: Text) -> JSONResponse: 
#     raw_text = text.text
#     language = prediction(raw_text)
#     return JSONResponse({"Language": language})


# if __name__ == "__main__": 
#     uvicorn.run(app, host="0.0.0.0", port=8000)
