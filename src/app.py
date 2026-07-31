"""FastAPI service exposing the two endpoints the evaluation harness calls.

    GET  /health -> 200 once the datasets are warm (hard gate for the run)
    POST /query  -> {"answer": ..., "steps": ..., "tool_trace": [...]}

Datasets are loaded during startup so the first graded question does not pay
the warm-up cost, and so /health only starts answering once the agent can
actually work. Request handling runs in a worker thread, keeping the event loop
free for the three concurrent requests the harness sends.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import afr_index, data_asx, data_rba
from .config import get_settings
from .graph import AgentGraph
from .schemas import HealthResponse, QueryRequest, QueryResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("agent")

STATE: dict[str, object] = {"warm": False, "graph": None}


def _warm_up() -> None:
    settings = get_settings()
    started = time.monotonic()

    data_rba.load()
    data_asx.load()
    afr_index.ensure_built(verbose=True)
    index = afr_index.get_index()
    index.normalised_headlines()

    STATE["graph"] = AgentGraph(settings)
    STATE["warm"] = True
    log.info(
        "warm in %.1fs | rba=%d asx=%d afr=%d | domain_predict_mode=%s brain=%s domain=%s",
        time.monotonic() - started,
        len(data_rba.load()),
        len(data_asx.load()),
        index.document_count,
        settings.domain_predict_mode,
        settings.brain_model,
        settings.domain_ft_model,
    )
    if not settings.uses_fine_tuned_model:
        log.warning(
            "DOMAIN_PREDICT_MODE=%s - answers are composed deterministically. "
            "Set DOMAIN_PREDICT_MODE=llm before official evaluation so the "
            "fine-tuned Nemotron model writes the answer.",
            settings.domain_predict_mode,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _warm_up()
    except Exception:
        log.exception("warm-up failed; serving in degraded mode")
        STATE["graph"] = AgentGraph(get_settings())
    yield


app = FastAPI(title="Cognitivo market-signal agent", version="1.0.0", lifespan=lifespan)


def _log_event(payload: dict) -> None:
    settings = get_settings()
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        path = settings.log_dir / f"agent-{datetime.now(timezone.utc):%Y%m%d}.jsonl"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:  # logging must never break a graded request
        log.exception("failed to write run log")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        datasets_loaded=bool(STATE["warm"]),
        domain_predict_mode=settings.domain_predict_mode,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> JSONResponse:
    request_id = uuid.uuid4().hex[:12]
    started = time.monotonic()
    graph: AgentGraph = STATE["graph"] or AgentGraph(get_settings())

    try:
        response, state = await asyncio.to_thread(graph.run, request.question)
        latency = time.monotonic() - started
        _log_event(
            {
                "request_id": request_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "question": request.question,
                "answer": response.answer,
                "latency_s": round(latency, 3),
                "steps": response.steps,
                "brain_turns": state.brain_turns,
                "tool_calls": [
                    {"tool": entry.tool, "args": entry.args} for entry in response.tool_trace
                ],
                "synth_mode": state.synth_mode,
                "used_fallback_planner": state.used_fallback_planner,
                "brain_error": state.brain_error,
                "path": state.path,
            }
        )
        if latency > 60:
            log.warning("request %s took %.1fs - over the 60s penalty threshold", request_id, latency)
        return JSONResponse(content=response.model_dump())

    except Exception as exc:
        # The contract requires valid JSON with a non-empty answer, always.
        log.exception("request %s failed", request_id)
        _log_event(
            {
                "request_id": request_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "question": request.question,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_s": round(time.monotonic() - started, 3),
            }
        )
        return JSONResponse(
            content=QueryResponse(
                answer=(
                    "The agent could not complete this question because of an internal "
                    "error, so no figure from the supplied datasets can be stated."
                ),
                steps=0,
                tool_trace=[],
            ).model_dump()
        )


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.agent_host, port=settings.agent_port, log_level="info")


if __name__ == "__main__":
    main()
