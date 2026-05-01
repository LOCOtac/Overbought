from fastapi import FastAPI
from overbought_oversold_tool import analyze_overbought_oversold

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Trading Tools API running"}

@app.get("/analyze")
def analyze(symbol: str):
    return analyze_overbought_oversold(symbol)
