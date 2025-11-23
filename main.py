from fastapi import FastAPI
import assignment4.api as assignment4_api

app = FastAPI(title="SPS GenAI - Assignment 4",
              description="Server for Assignment 4: Diffusion and Energy-Based Models")

# Mount the assignment4 app at root so graders can reach docs at /docs
app.mount("/", assignment4_api.app)

# Helpful root message (also not shown in schema)
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Assignment4 app mounted at root. Open /docs for API documentation. Endpoints: /generate/diffusion , /generate/ebm"
    }
