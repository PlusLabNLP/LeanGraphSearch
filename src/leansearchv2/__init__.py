"""LeanSearch v2 package root.

Lightweight by default: importing this module does not pull in cuVS or torch.
Import `leansearchv2.pipeline` explicitly when you need the GPU retriever
(i.e. inside the server process).
"""

from .standard_client import ResultData, SearchResult, StandardClient
from .llm import LLMClient

__all__ = ["ResultData", "SearchResult", "StandardClient", "LLMClient"]
