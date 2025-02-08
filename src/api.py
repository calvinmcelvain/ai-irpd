from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from irpd import IRPD
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IRPDRequest(BaseModel):
    case: str
    ras: List[str]
    treatments: List[str]
    stages: List[str]
    test_type: Optional[str] = "test"
    llms: Optional[List[str]] = ["GPT_4O_1120"]
    llm_config: Optional[str] = "base"
    N: Optional[int] = 1
    max_instances: Optional[int] = None
    project_path: Optional[str] = None
    print_response: Optional[bool] = False
    new_test: Optional[bool] = True


@app.post("/run_irpd_test")
def run_irpd_test(request: IRPDRequest):
    try:
        irpd_test = IRPD(**request.model_dump())
        irpd_test.run()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
