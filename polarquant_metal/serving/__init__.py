"""
DEPRECATED: The serving layer has moved to the `polarquant-serving` package.
    pip install polarquant-serving
    python -m polarquant_serving.server --model ...

This copy is kept for backward compatibility.
"""

from .server import create_app, serve

__all__ = ["create_app", "serve"]
