from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from graph import workflow

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/submit_stream")
async def submit_stream(message: str = Form("")):
    return {"workflow": workflow is not None, "message": message}
