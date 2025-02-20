from fastapi import FastAPI
from pydantic import BaseModel
from Agent_controller import agentcontroller

# Initialize FastAPI app
app = FastAPI()

# Create a Pydantic model for request body
class MessageRequest(BaseModel):
    input: dict

# Initialize the agent controller outside the route handler for efficiency
Bot_agent = agentcontroller()

@app.post("/get-response")
async def get_response(request: MessageRequest):
    response = Bot_agent.get_response(request.dict())
   
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
