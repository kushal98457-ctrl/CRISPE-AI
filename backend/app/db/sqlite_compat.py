"""SQLite dialect compatibility helper for pgvector vector operations in tests and development."""

import json
import math
from typing import Any, Sequence, Union
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.elements import BinaryExpression


@compiles(BinaryExpression, "sqlite")
def compile_binary_sqlite(binary: BinaryExpression, compiler: Any, **kw: Any) -> str:
    """Compile pgvector '<=>' cosine distance operator to SQLite custom function 'cosine_distance'."""
    op = getattr(binary, "operator", None)
    if hasattr(op, "opstring") and op.opstring == "<=>":
        return f"cosine_distance({compiler.process(binary.left, **kw)}, {compiler.process(binary.right, **kw)})"
    return compiler.visit_binary(binary, **kw)


def sqlite_cosine_distance(v1_raw: Any, v2_raw: Any) -> float:
    """Compute mathematical cosine distance between two vectors in SQLite: 1 - (dot / (|v1| * |v2|)).

    Cosine distance ranges from 0.0 (identical angle) to 2.0 (opposite directions).
    """
    if isinstance(v1_raw, str):
        try:
            v1 = json.loads(v1_raw)
        except Exception:
            v1 = [float(x.strip()) for x in v1_raw.strip("[]()").split(",") if x.strip()]
    else:
        v1 = v1_raw

    if isinstance(v2_raw, str):
        try:
            v2 = json.loads(v2_raw)
        except Exception:
            v2 = [float(x.strip()) for x in v2_raw.strip("[]()").split(",") if x.strip()]
    else:
        v2 = v2_raw

    if not v1 or not v2 or len(v1) != len(v2):
        return 1.0

    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    for a, b in zip(v1, v2):
        fa = float(a)
        fb = float(b)
        dot += fa * fb
        norm1_sq += fa * fa
        norm2_sq += fb * fb

    norm1 = math.sqrt(norm1_sq)
    norm2 = math.sqrt(norm2_sq)
    if norm1 == 0.0 or norm2 == 0.0:
        return 1.0

    cos_sim = dot / (norm1 * norm2)
    # Cosine distance = 1.0 - cosine_similarity, clamped between 0.0 and 2.0
    return max(0.0, min(2.0, 1.0 - cos_sim))


def register_sqlite_vector_compat(engine: Union[Engine, Any]) -> None:
    """Register custom SQLite cosine_distance function on an engine connection pool."""
    target_engine = getattr(engine, "sync_engine", engine)
    if hasattr(target_engine, "dialect") and target_engine.dialect.name == "sqlite":
        @event.listens_for(target_engine, "connect")
        def _on_sqlite_connect(dbapi_connection: Any, connection_record: Any) -> None:
            if hasattr(dbapi_connection, "create_function"):
                dbapi_connection.create_function("cosine_distance", 2, sqlite_cosine_distance)
