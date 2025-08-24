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
from .registry import ModelRegistry
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
        self.registry = ModelRegistry()  # Dynamic runtime registry
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

            # Get models from registry
            for port, model_entry in self.registry.get_all_models().items():
                models_status.append(
                    ModelStatus(
                        name=model_entry.display_name,
                        model_path="",  # Not tracked in registry
                        port=port,
                        gpu_ids=model_entry.gpu_ids,
                        status=(
                            "running"
                            if model_entry.state.value == "running"
                            else "stopped"
                        ),
                        registration_status=model_entry.status.value,
                        request_count=0,  # Not tracked in simplified registry
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

        @self.app.post("/proxy/pre_register")
        async def pre_register(request: Dict[str, Any]):
            """Pre-register a model with pending status."""
            try:
                port = request["port"]
                gpu_ids = request.get("gpu_ids", [])
                config_name = request.get("config_name")
                request.get("estimated_util")

                logger.info(
                    f"Pre-registering model '{config_name}' on port {port} "
                    f"with GPUs {gpu_ids}"
                )

                # Pre-register in the runtime registry
                success = self.registry.pre_register(port, gpu_ids, config_name)

                if success:
                    logger.info(f"Pre-registered model on port {port}")
                    return {
                        "status": "success",
                        "message": f"Pre-registered model on port {port}",
                    }
                else:
                    message = f"Failed to pre-register model on port {port}"
                    logger.error(message)
                    raise HTTPException(status_code=400, detail=message)

            except Exception as e:
                logger.error(f"Pre-registration error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/proxy/register")
        async def register_model(request: Dict[str, Any]):
            """Register a new model in the runtime registry (backward compatibility)."""
            try:
                port = request["port"]
                gpu_ids = request.get("gpu_ids", [])
                actual_name = request.get("actual_name", f"model_{port}")

                # Register in the runtime registry
                success = self.registry.verify_and_activate(port, actual_name)

                if success:
                    # Add to router for request routing
                    backend_url = f"http://localhost:{port}"
                    self.router.add_backend(
                        actual_name, backend_url, {"port": port, "gpu_ids": gpu_ids}
                    )

                    return {
                        "status": "success",
                        "message": f"Registered model '{actual_name}' on port {port}",
                        "actual_name": actual_name,
                    }
                else:
                    message = f"Failed to register model on port {port}"
                    raise HTTPException(status_code=400, detail=message)

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.delete("/proxy/models/{port}")
        async def unregister_model(port: int):
            """Unregister a model from the runtime registry."""
            try:
                # Get model info before unregistering
                model_entry = self.registry.get_model(port)
                if not model_entry:
                    raise HTTPException(
                        status_code=404, detail=f"Model on port {port} not found"
                    )

                # Unregister from registry
                success = self.registry.remove_model(port)
                message = (
                    f"Model on port {port} unregistered"
                    if success
                    else f"Model on port {port} not found"
                )

                if success:
                    # Remove from router
                    if model_entry.actual_name:
                        try:
                            self.router.remove_backend(model_entry.actual_name)
                        except Exception:
                            pass  # Router entry might not exist

                    return {"status": "success", "message": message}
                else:
                    raise HTTPException(status_code=400, detail=message)

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/proxy/state")
        async def update_model_state(request: Dict[str, Any]):
            """Update the state of a model (running/sleeping/stopped)."""
            try:
                port = request["port"]
                state = request["state"]
                sleep_level = request.get("sleep_level", 0)

                # Update model state
                model_entry = self.registry.get_model(port)
                if model_entry:
                    from .registry import ModelState

                    if state == "sleeping":
                        model_entry.state = ModelState.SLEEPING
                        model_entry.sleep_level = sleep_level
                    elif state == "running":
                        model_entry.state = ModelState.RUNNING
                        model_entry.sleep_level = 0
                    elif state == "stopped":
                        model_entry.state = ModelState.STOPPED
                    message = f"Updated state to {state}"
                    return {"status": "success", "message": message}
                else:
                    raise HTTPException(
                        status_code=404, detail=f"Model on port {port} not found"
                    )

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/proxy/registry")
        async def get_registry():
            """Get the complete registry status including GPU allocation."""
            return self.registry.get_status_summary()

        @self.app.post("/proxy/refresh_models")
        async def refresh_models():
            """
            Refresh model registry - verify ALL models and update their status.
            """
            import httpx

            from .registry import ModelState, RegistrationStatus

            # Cleanup any stale entries
            removed_count = self.registry.cleanup_stale_entries()
            if removed_count > 0:
                logger.info(f"Removed {removed_count} stale entries")

            # Get all models and check their status
            all_models = self.registry.get_all_models()
            newly_registered = []
            already_available = []
            newly_failed = []
            pending_count = 0

            for port, model_entry in all_models.items():
                # Check ALL models
                try:
                    # Call the model's /v1/models endpoint
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{port}/v1/models", timeout=5.0
                        )

                        if response.status_code == 200:
                            # Model is responding
                            # Always check if model is sleeping
                            # (not just when marked as sleeping)
                            is_sleeping = False
                            try:
                                sleep_response = await client.get(
                                    f"http://localhost:{port}/is_sleeping", timeout=2.0
                                )
                                if sleep_response.status_code == 200:
                                    sleep_data = sleep_response.json()
                                    is_sleeping = sleep_data.get("is_sleeping", False)

                                    # Update model state based on actual sleep status
                                    if (
                                        is_sleeping
                                        and model_entry.state != ModelState.SLEEPING
                                    ):
                                        model_entry.state = ModelState.SLEEPING
                                        logger.info(f"Model on port {port} is sleeping")
                                    elif (
                                        not is_sleeping
                                        and model_entry.state == ModelState.SLEEPING
                                    ):
                                        model_entry.state = ModelState.RUNNING
                                        logger.info(
                                            f"Model on port {port} has been woken up"
                                        )
                                    elif not is_sleeping:
                                        model_entry.state = ModelState.RUNNING
                            except Exception:
                                # If /is_sleeping endpoint doesn't exist or fails,
                                # check current state. If it was sleeping, keep it as
                                # sleeping; otherwise assume running
                                if model_entry.state != ModelState.SLEEPING:
                                    model_entry.state = ModelState.RUNNING

                            # Extract actual model name from response
                            models_data = response.json()
                            actual_name = None
                            if models_data.get("data"):
                                actual_name = models_data["data"][0].get("id")

                            # Check previous status
                            was_pending = (
                                model_entry.status == RegistrationStatus.PENDING
                            )
                            was_error = model_entry.status == RegistrationStatus.ERROR

                            # Verify and activate the model
                            if self.registry.verify_and_activate(port, actual_name):
                                if was_pending or was_error:
                                    newly_registered.append(
                                        actual_name or f"port_{port}"
                                    )
                                else:
                                    already_available.append(
                                        actual_name or f"port_{port}"
                                    )

                                # Ensure it's in the router
                                if (
                                    actual_name
                                    and actual_name not in self.router.backends
                                ):
                                    backend_url = f"http://localhost:{port}"
                                    self.router.add_backend(
                                        actual_name,
                                        backend_url,
                                        {
                                            "port": port,
                                            "gpu_ids": model_entry.gpu_ids,
                                        },
                                    )
                                logger.debug(
                                    f"Model on port {port} verified as available"
                                )
                        else:
                            # Model not responding properly
                            if model_entry.status == RegistrationStatus.AVAILABLE:
                                # Was available, now failing
                                model_entry.mark_error(f"HTTP {response.status_code}")
                                newly_failed.append(model_entry.display_name)
                                # Remove from router if present
                                if model_entry.actual_name in self.router.backends:
                                    self.router.remove_backend(model_entry.actual_name)
                            else:
                                pending_count += 1

                except Exception as e:
                    # Model not accessible
                    if model_entry.status == RegistrationStatus.AVAILABLE:
                        # Was available, now failing
                        model_entry.mark_error(
                            str(e)[:100]
                        )  # Limit error message length
                        newly_failed.append(model_entry.display_name)
                        # Remove from router if present
                        if (
                            model_entry.actual_name
                            and model_entry.actual_name in self.router.backends
                        ):
                            self.router.remove_backend(model_entry.actual_name)
                        logger.warning(
                            f"Model on port {port} no longer accessible: {e}"
                        )
                    elif model_entry.status == RegistrationStatus.PENDING:
                        pending_count += 1
                        logger.debug(f"Model on port {port} still not ready: {e}")

            summary = {
                "registered": len(newly_registered),
                "already_available": len(already_available),
                "failed": len(newly_failed),
                "pending": pending_count,
                "removed": removed_count,
                "total": len(all_models),
            }

            return {
                "status": "success",
                "summary": summary,
                "details": {
                    "newly_registered": newly_registered,
                    "already_available": already_available,
                    "newly_failed": newly_failed,
                },
            }

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
