"""growthagent — an autonomous growth agent (AI Agent engineering demo).

A minimal, dependency-light implementation of an LLM agent that plans,
calls tools, observes results and reflects — aimed at the "autonomous
digital worker" use case (ad creative + campaign + optimization).

Core runtime dependency: numpy only.  Everything else (OpenAI / Ollama /
FastAPI / gRPC / tiktoken / hnswlib) is optional and lazy-imported.
"""

__version__ = "0.1.0"
