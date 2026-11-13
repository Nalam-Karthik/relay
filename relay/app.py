"""HTTP layer: an OpenAI-compatible API in front of the gateway pipeline."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .config import Settings, load_settings
from .gateway import Gateway, GatewayError
from .ledger import month_start
from .providers import Provider, ProviderError, StreamDelta, build_providers
from .schemas import ChatCompletionRequest


def _error_body(code: str, message: str, request_id: str) -> dict:
    return {"error": {"type": code, "code": code, "message": message, "request_id": request_id}}


def _sse(payload: dict | str) -> str:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n"


def create_app(settings: Settings | None = None, providers: dict[str, Provider] | None = None) -> FastAPI:
    settings = settings or load_settings()
    gateway = Gateway(settings, providers if providers is not None else build_providers(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await gateway.aclose()

    app = FastAPI(title="Relay LLM Gateway", version="0.1.0", lifespan=lifespan)
    app.state.gateway = gateway

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError):
        return JSONResponse(
            _error_body(exc.code, exc.message, request.state.request_id), status_code=exc.status, headers=exc.headers
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest, request: Request, authorization: str | None = Header(default=None)
    ):
        rid = request.state.request_id
        tenant = gateway.authenticate(authorization)
        created = int(time.time())
        completion_id = f"chatcmpl-{rid}"

        if body.stream:
            result = await gateway.stream(tenant, body, rid)

            async def sse():
                def chunk(delta: dict, finish: str | None = None, **extra) -> str:
                    return _sse({
                        "id": completion_id, "object": "chat.completion.chunk", "created": created,
                        "model": body.model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra,
                    })

                yield chunk({"role": "assistant"})
                try:
                    async for event in result.events:
                        if isinstance(event, StreamDelta):
                            yield chunk({"content": event.text})
                        else:
                            usage = {
                                "prompt_tokens": event.input_tokens,
                                "completion_tokens": event.output_tokens,
                                "total_tokens": event.input_tokens + event.output_tokens,
                            }
                            yield chunk({}, event.finish_reason, usage=usage)
                except ProviderError as e:
                    yield _sse(_error_body("upstream_error", str(e), rid))
                yield _sse("[DONE]")

            headers = {
                "x-relay-provider": result.provider,
                "x-relay-model": result.model,
                "x-relay-cache": result.cache,
                "x-relay-attempts": str(len(result.attempts) + 1),
                "cache-control": "no-cache",
            }
            return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)

        outcome = await gateway.complete(tenant, body, rid)
        c = outcome.completion
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": body.model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": c.text}, "finish_reason": c.finish_reason}
                ],
                "usage": {
                    "prompt_tokens": c.input_tokens,
                    "completion_tokens": c.output_tokens,
                    "total_tokens": c.input_tokens + c.output_tokens,
                },
            },
            headers={
                "x-relay-provider": c.provider,
                "x-relay-model": c.model,
                "x-relay-cache": outcome.cache,
                "x-relay-attempts": str(len(outcome.attempts) + 1),
            },
        )

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        tenant = gateway.authenticate(authorization)
        names = tenant.allowed_models if tenant.allowed_models is not None else list(settings.routes)
        return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "relay"} for n in names]}

    @app.get("/v1/usage")
    async def usage(authorization: str | None = Header(default=None)):
        tenant = gateway.authenticate(authorization)
        summary = gateway.ledger.summary(tenant.name, month_start())
        return {
            "tenant": tenant.name,
            "period": "month_to_date",
            "monthly_budget_usd": tenant.monthly_budget_usd,
            **summary,
        }

    @app.get("/admin/providers")
    async def provider_status(authorization: str | None = Header(default=None)):
        tenant = gateway.authenticate(authorization)
        if not tenant.admin:
            raise GatewayError(403, "forbidden", "admin key required")
        return gateway.provider_status()

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return gateway.metrics.render()

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
