"""Request context middleware for correlation IDs and processing telemetry."""

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from app.core.logging import request_id_ctx


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware capturing request ID and execution duration."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Honor client-provided X-Request-ID or generate a clean UUID4 identifier
        incoming_req_id = request.headers.get("X-Request-ID")
        if incoming_req_id and len(incoming_req_id) <= 64:
            request_id = incoming_req_id
        else:
            request_id = f"req-{uuid.uuid4().hex[:12]}"

        # Bind to context variable for logging
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            process_time = time.perf_counter() - start_time
            request_id_ctx.reset(token)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        return response
