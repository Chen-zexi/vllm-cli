#!/usr/bin/env python3
"""
Data models for the proxy server.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single model in the proxy."""

    name: str = Field(..., description="Model identifier for API requests")
    model_path: str = Field(..., description="Path or HuggingFace model ID")
    gpu_ids: List[int] = Field(default_factory=list, description="GPU IDs to use")
    port: int = Field(..., description="Port for the vLLM server")
    profile: Optional[str] = Field(None, description="vLLM CLI profile to use")
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Additional vLLM configuration"
    )
    enabled: bool = Field(True, description="Whether model is enabled")


class ProxyConfig(BaseModel):
    """Configuration for the proxy server."""

    host: str = Field("0.0.0.0", description="Proxy server host")  # nosec B104
    port: int = Field(8000, description="Proxy server port")
    models: List[ModelConfig] = Field(
        default_factory=list, description="List of models to serve"
    )
    enable_cors: bool = Field(True, description="Enable CORS headers")
    enable_metrics: bool = Field(True, description="Enable metrics endpoint")
    log_requests: bool = Field(False, description="Log all requests")


class ModelStatus(BaseModel):
    """Status information for a model."""

    name: str
    model_path: str
    port: int
    gpu_ids: List[int]
    status: str  # "running", "starting", "stopped", "error"
    registration_status: Optional[str] = None  # "pending", "available", "error"
    uptime: Optional[float] = None
    error_message: Optional[str] = None
    request_count: int = 0
    last_request_time: Optional[str] = None


class ProxyStatus(BaseModel):
    """Overall proxy server status."""

    proxy_running: bool
    proxy_port: int
    proxy_host: str
    models: List[ModelStatus]
    total_requests: int = 0
    start_time: Optional[str] = None
