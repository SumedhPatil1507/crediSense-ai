"""Small OpenTelemetry helpers shared by the RAG and scoring paths.

The API remains usable when telemetry is not configured; installing
opentelemetry-api/sdk enables real spans and SDK exporters can be configured by
the host application.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    tracer = trace.get_tracer("credisense.credit-risk")
except ImportError:  # pragma: no cover - exercised on minimal installations
    class _NoopSpan:
        def set_attribute(self, *_: Any, **__: Any) -> None:
            pass

        def record_exception(self, *_: Any, **__: Any) -> None:
            pass

        def set_status(self, *_: Any, **__: Any) -> None:
            pass

    @contextmanager
    def _noop_span(*_: Any, **__: Any) -> Iterator[_NoopSpan]:
        yield _NoopSpan()

    class _NoopTracer:
        start_as_current_span = staticmethod(_noop_span)

    class _StatusCode:
        ERROR = "ERROR"

    class _Status:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

    tracer = _NoopTracer()
    Status = _Status
    StatusCode = _StatusCode


@contextmanager
def traced_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span, attach safe attributes, and record failures before re-raising."""
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


def traced_predict_proba(model: Any, features: Any, *, operation: str = "predict") -> Any:
    """Trace an inference call without recording feature values or identifiers."""
    with traced_span(
        "credit_risk.score",
        **{"credit_risk.operation": operation, "credit_risk.model": type(model).__name__},
    ):
        return model.predict_proba(features)
