from typing import Literal

from pydantic import BaseModel


class ComponentHealth(BaseModel):
    configuration: Literal["ok"] = "ok"
    gemini: Literal["configured", "not_configured"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    components: ComponentHealth
    request_id: str


class ProblemDetail(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    code: str
    request_id: str

