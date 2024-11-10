from pydantic import BaseModel

class QueryResponse(BaseModel):
    response: dict