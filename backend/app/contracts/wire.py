"""snake_case (Python side, see contracts/__init__.py docstring) <-> camelCase
(the wire format the already-built React frontend expects, since its
src/api/socket.ts does no case conversion of its own) conversion, applied only
at the API boundary -- never baked into the contract models themselves."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

_UNDERSCORE_RE = re.compile(r"_([a-zA-Z0-9])")


def snake_to_camel(name: str) -> str:
    return _UNDERSCORE_RE.sub(lambda m: m.group(1).upper(), name)


def camelize(obj: Any) -> Any:
    """Recursively convert dict keys from snake_case to camelCase."""
    if isinstance(obj, dict):
        return {snake_to_camel(k): camelize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [camelize(v) for v in obj]
    return obj


def to_wire(model: BaseModel) -> dict:
    """Serialize a contract model to the camelCase dict shape the frontend
    expects. JSON-mode dump first so dates/enums are already wire-primitive."""
    return camelize(model.model_dump(mode="json", exclude_none=True))
