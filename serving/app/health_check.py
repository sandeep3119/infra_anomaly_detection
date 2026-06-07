from fastapi import Request,HTTPException,APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness_check():
    return {
        "status": "Ok"
    }

@router.get("/ready")
async def rediness_check(request:Request):
    if getattr(request.app.state, "model", None) is None:
        raise HTTPException(status_code=503,detail="App not ready, as model is not available")
    return {
        "status": "Ok"
    }

    
