from pathlib import Path

from trpg_agent.langchain.models import infer_provider, load_model_config


def test_infer_provider() -> None:
    assert infer_provider("https://api.example.com/anthropic") == "anthropic-compatible"
    assert infer_provider("https://api.example.com/v1") == "openai-compatible"


def test_load_model_config(tmp_path: Path) -> None:
    path = tmp_path / "llm.config.json"
    path.write_text(
        '{"apiKey":"key","baseURL":"https://api.example.com/anthropic","model":"m"}',
        encoding="utf-8",
    )

    config = load_model_config(path)

    assert config.provider == "anthropic-compatible"
    assert config.model == "m"
    assert config.api_key == "key"
