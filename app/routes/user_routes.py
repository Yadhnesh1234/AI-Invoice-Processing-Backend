from fastapi import APIRouter, HTTPException, Depends
from models.user import User
from db.database import database
from bson import ObjectId

router = APIRouter()

@router.post("/users/", response_model=User)
async def create_user(user: User):
    user_dict = user.dict()
    result = await database["users"].insert_one(user_dict)
    user_dict["_id"] = str(result.inserted_id)
    return user_dict

@router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = await database["users"].find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["_id"] = str(user["_id"])
    return user
