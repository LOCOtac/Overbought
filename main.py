from fastapi import FastAPI
from overbought_oversold_tool import analyze_overbought_oversold
from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.get("/analyze")
def analyze(symbol: str):
    return analyze_overbought_oversold(symbol)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")