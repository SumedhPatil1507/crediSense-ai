"""Small OpenTelemetry helpers shared by the RAG and scoring paths.

The API remains usable when telemetry is not configured; installing
opentelemetry-api/sdk enables real spans and SDK exporters can be configured by
the host application.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from typing import Any, Iterator

LOGGER = logging.getLogger(__name__)

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
    """Trace a probability inference without recording features or identifiers."""
    return traced_model_call(model, "predict_proba", features, operation=operation)


def traced_model_call(model: Any, method: str, features: Any, *,
                      operation: str = "inference") -> Any:
    """Trace a model method call without attaching input values or identifiers."""
    if method.startswith("_") or not callable(getattr(model, method, None)):
        raise ValueError(f"Unsupported model inference method: {method}")
    with traced_span(
        "model.inference",
        **{"model.operation": operation, "model.name": type(model).__name__,
           "model.method": method},
    ):
        return getattr(model, method)(features)


def configure_telemetry() -> bool:
    """Install an SDK tracer provider and optional OTLP span exporter once.

    The API works without the SDK. Set OTEL_EXPORTER_OTLP_ENDPOINT (or the
    traces-specific endpoint) to export spans to a Collector/backend.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return False

    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return True
    resource = Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "credisense-api")})
    provider = TracerProvider(resource=resource)
    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or
                os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
    if endpoint:
        protocol = (os.getenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") or
                    os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")).lower()
        try:
            if protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            elif protocol == "http/protobuf":
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            else:
                raise ValueError("OTEL OTLP traces protocol must be grpc or http/protobuf")
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except (ImportError, ValueError):
            LOGGER.exception("OTLP trace exporter could not be configured")
    trace.set_tracer_provider(provider)
    return True


configure_telemetry()
