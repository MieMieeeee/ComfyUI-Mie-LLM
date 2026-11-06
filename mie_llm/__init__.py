"""Mie's LLM client layer.

Every provider connector is an HTTP client for a chat endpoint; the package
never loads a model itself. That is what keeps the dependency surface at
``requests``.
"""

__version__ = "0.1.4"
