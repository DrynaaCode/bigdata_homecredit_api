from pydantic import BaseModel

class StrictBase(BaseModel):
    class Config:
        extra = "forbid"
        orm_mode = True
