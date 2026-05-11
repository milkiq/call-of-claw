from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import SecretStr

ModelProvider = Literal["anthropic-compatible", "openai-compatible"]


@dataclass(frozen=True)
class ModelConfig:
    provider: ModelProvider
    model: str
    api_key: str
    base_url: str
    temperature: float = 0
    timeout_seconds: int = 60


def describe_model(config: ModelConfig) -> dict[str, str | float | int]:
    """Return metadata used in LangSmith traces.

    Provider construction is intentionally deferred until API credentials are configured.
    """

    return {
        "model_provider": config.provider,
        "model_name": config.model,
        "temperature": config.temperature,
        "timeout_seconds": config.timeout_seconds,
    }


def infer_provider(base_url: str) -> ModelProvider:
    lowered = base_url.lower()
    if "anthropic" in lowered:
        return "anthropic-compatible"
    return "openai-compatible"


def load_model_config(path: Path = Path("llm.config.json")) -> ModelConfig:
    if not path.exists():
        raise FileNotFoundError(f"Missing LLM config: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    api_key = raw.get("apiKey") or raw.get("api_key")
    base_url = raw.get("baseURL") or raw.get("base_url")
    model = raw.get("model")
    if not api_key or not base_url or not model:
        raise ValueError("LLM config must include apiKey, baseURL, and model")
    provider = raw.get("provider") or infer_provider(str(base_url))
    if provider not in {"anthropic-compatible", "openai-compatible"}:
        raise ValueError(f"Unsupported model provider: {provider}")
    return ModelConfig(
        provider=provider,
        model=str(model),
        api_key=str(api_key),
        base_url=str(base_url).rstrip("/"),
        temperature=float(raw.get("temperature", 0)),
        timeout_seconds=int(raw.get("timeoutSeconds", raw.get("timeout_seconds", 60))),
    )


def build_chat_model(config: ModelConfig) -> BaseChatModel:
    if config.provider == "anthropic-compatible":
        return AnthropicCompatibleChatModel(
            model=config.model,
            api_key=SecretStr(config.api_key),
            base_url=config.base_url,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
        )
    return OpenAICompatibleChatModel(
        model=config.model,
        api_key=SecretStr(config.api_key),
        base_url=config.base_url,
        temperature=config.temperature,
        timeout_seconds=config.timeout_seconds,
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


class AnthropicCompatibleChatModel(BaseChatModel):
    model: str
    api_key: SecretStr
    base_url: str
    temperature: float = 0
    timeout_seconds: int = 60
    max_tokens: int = 2000

    @property
    def _llm_type(self) -> str:
        return "anthropic-compatible"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            text = _content_to_text(message.content)
            if isinstance(message, SystemMessage):
                system_parts.append(text)
            elif isinstance(message, AIMessage):
                anthropic_messages.append({"role": "assistant", "content": text})
            elif isinstance(message, HumanMessage):
                anthropic_messages.append({"role": "user", "content": text})
            else:
                anthropic_messages.append({"role": "user", "content": text})

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": self.temperature,
            "messages": anthropic_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if stop:
            payload["stop_sequences"] = stop

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key.get_secret_value(),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        text = _content_to_text(data.get("content", ""))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class OpenAICompatibleChatModel(BaseChatModel):
    model: str
    api_key: SecretStr
    base_url: str
    temperature: float = 0
    timeout_seconds: int = 60
    max_tokens: int = 2000

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "messages": [
                {
                    "role": "system"
                    if isinstance(message, SystemMessage)
                    else "assistant"
                    if isinstance(message, AIMessage)
                    else "user",
                    "content": _content_to_text(message.content),
                }
                for message in messages
            ],
        }
        if stop:
            payload["stop"] = stop

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "authorization": f"Bearer {self.api_key.get_secret_value()}",
                    "content-type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        text = data["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
