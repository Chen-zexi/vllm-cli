#!/usr/bin/env python3
"""
FastAPI-based proxy server for routing requests to multiple vLLM instances.
"""
import json
import logging
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .models import ModelStatus, ProxyConfig, ProxyStatus
from .router import RequestRouter

logger = logging.getLogger(__name__)


class ProxyServer:
    """
    FastAPI proxy server that routes OpenAI API requests to appropriate vLLM instances.
    """

    def __init__(self, config: ProxyConfig):
        """Initialize the proxy server with configuration."""
        self.config = config
        self.app = FastAPI(title="vLLM Multi-Model Proxy", version="0.1.0")
        self.router = RequestRouter()
        self.start_time = datetime.now()
        self.total_requests = 0
        self.model_requests: Dict[str, int] = {}

        # HTTP client for forwarding requests
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=300.0))

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Configure middleware for the FastAPI app."""
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Request logging middleware
        if self.config.log_requests:

            @self.app.middleware("http")
            async def log_requests(request: Request, call_next):
                start_time = time.time()
                response = await call_next(request)
                duration = time.time() - start_time
                logger.info(
                    f"{request.method} {request.url.path} - "
                    f"{response.status_code} - {duration:.3f}s"
                )
                return response

    def _setup_routes(self):
        """Setup API routes for the proxy server."""

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "message": "vLLM Multi-Model Proxy Server",
                "version": "0.1.0",
                "models_count": len(self.router.get_active_models()),
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.get("/v1/models")
        async def list_models():
            """List available models (OpenAI API compatible)."""
            models = self.router.get_active_models()
            return {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "vllm-cli-proxy",
                    }
                    for model_name in models
                ],
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            """Handle chat completions requests."""
            return await self._forward_request(request, "/v1/chat/completions")

        @self.app.post("/v1/completions")
        async def completions(request: Request):
            """Handle completions requests."""
            return await self._forward_request(request, "/v1/completions")

        @self.app.post("/v1/embeddings")
        async def embeddings(request: Request):
            """Handle embeddings requests."""
            return await self._forward_request(request, "/v1/embeddings")

        @self.app.get("/proxy/status")
        async def proxy_status():
            """Get proxy server status."""
            models_status = []
            for model_name, backend in self.router.get_backends().items():
                models_status.append(
                    ModelStatus(
                        name=model_name,
                        model_path=backend.get("model_path", ""),
                        port=backend["port"],
                        gpu_ids=backend.get("gpu_ids", []),
                        status=await self._check_backend_health(backend["port"]),
                        request_count=self.model_requests.get(model_name, 0),
                    )
                )

            return ProxyStatus(
                proxy_running=True,
                proxy_port=self.config.port,
                proxy_host=self.config.host,
                models=models_status,
                total_requests=self.total_requests,
                start_time=self.start_time.isoformat(),
            )

        @self.app.post("/proxy/add_model")
        async def add_model(model_config: Dict[str, Any]):
            """Dynamically add a new model to the proxy."""
            try:
                model_name = model_config["name"]
                backend_url = f"http://localhost:{model_config['port']}"
                self.router.add_backend(model_name, backend_url, model_config)
                return {"status": "success", "message": f"Model {model_name} added"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.delete("/proxy/remove_model/{model_name}")
        async def remove_model(model_name: str):
            """Remove a model from the proxy."""
            try:
                self.router.remove_backend(model_name)
                return {"status": "success", "message": f"Model {model_name} removed"}
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))

        if self.config.enable_metrics:

            @self.app.get("/metrics")
            async def metrics():
                """Prometheus-compatible metrics endpoint."""
                metrics_text = []
                metrics_text.append(
                    "# HELP proxy_requests_total Total requests to proxy"
                )
                metrics_text.append("# TYPE proxy_requests_total counter")
                metrics_text.append(f"proxy_requests_total {self.total_requests}")

                metrics_text.append("# HELP model_requests_total Requests per model")
                metrics_text.append("# TYPE model_requests_total counter")
                for model, count in self.model_requests.items():
                    metrics_text.append(
                        f'model_requests_total{{model="{model}"}} {count}'
                    )

                uptime = (datetime.now() - self.start_time).total_seconds()
                metrics_text.append(
                    "# HELP proxy_uptime_seconds Proxy uptime in seconds"
                )
                metrics_text.append("# TYPE proxy_uptime_seconds gauge")
                metrics_text.append(f"proxy_uptime_seconds {uptime}")

                return "\n".join(metrics_text)

    async def _forward_request(self, request: Request, endpoint: str):
        """
        Forward a request to the appropriate vLLM backend.

        Args:
            request: Incoming FastAPI request
            endpoint: Target endpoint path

        Returns:
            Response from the backend server
        """
        try:
            # Parse request body
            body = await request.body()
            json_body = json.loads(body) if body else {}

            # Extract model from request
            model_name = json_body.get("model")
            if not model_name:
                raise HTTPException(
                    status_code=400, detail="Missing 'model' field in request"
                )

            # Get backend URL for this model
            backend_url = self.router.route_request(model_name)
            if not backend_url:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model_name}' not found"
                )

            # Update request counters
            self.total_requests += 1
            self.model_requests[model_name] = self.model_requests.get(model_name, 0) + 1

            # Check if streaming is requested
            stream = json_body.get("stream", False)

            # Forward the request
            target_url = f"{backend_url}{endpoint}"
            headers = dict(request.headers)
            headers.pop("host", None)  # Remove host header

            if stream:
                # Handle streaming response
                return StreamingResponse(
                    self._stream_response(target_url, headers, body),
                    media_type="text/event-stream",
                )
            else:
                # Handle regular response
                response = await self.client.post(
                    target_url, headers=headers, content=body
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code, detail=response.text
                    )

                return JSONResponse(
                    content=response.json(), status_code=response.status_code
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error forwarding request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _stream_response(
        self, url: str, headers: Dict, body: bytes
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream response from backend server.

        Args:
            url: Target URL
            headers: Request headers
            body: Request body

        Yields:
            Chunks of response data
        """
        try:
            async with self.client.stream(
                "POST", url, headers=headers, content=body
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield f'data: {{"error": "{str(e)}"}}\n\n'.encode()

    async def _check_backend_health(self, port: int) -> str:
        """
        Check health of a backend server.

        Args:
            port: Backend server port

        Returns:
            Status string: "running", "error", or "unknown"
        """
        try:
            response = await self.client.get(
                f"http://localhost:{port}/health", timeout=5.0
            )
            return "running" if response.status_code == 200 else "error"
        except Exception:
            return "unknown"

    def run(self):
        """Run the proxy server."""
        import uvicorn

        logger.info(f"Starting proxy server on {self.config.host}:{self.config.port}")

        uvicorn.run(
            self.app, host=self.config.host, port=self.config.port, log_level="info"
        )

    async def shutdown(self):
        """Cleanup resources on shutdown."""
        await self.client.aclose()
