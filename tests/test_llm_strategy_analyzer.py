import os
import pytest
from llm_strategy_analyzer import LLMStrategyAnalyzer

# Helper to set and unset env vars
def set_env(key, value):
    os.environ[key] = value
def unset_env(key):
    if key in os.environ:
        del os.environ[key]

def test_llm_strategy_analyzer_initialization():
    providers = [
        ("openai", "OPENAI_API_KEY", "gpt-4-turbo-preview"),
        ("anthropic", "ANTHROPIC_API_KEY", "claude-3-5-sonnet-20241022"),
        ("google", "GOOGLE_API_KEY", "gemini-pro"),
        ("openrouter", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct"),
    ]
    for provider, env_var, default_model in providers:
        set_env(env_var, "dummy_key")
        analyzer = LLMStrategyAnalyzer(provider=provider)
        assert analyzer.provider == provider
        assert analyzer.model == default_model
        assert analyzer.api_key == "dummy_key"
        unset_env(env_var)

def test_llm_strategy_analyzer_missing_api_key():
    providers = [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ]
    for provider, env_var in providers:
        unset_env(env_var)
        with pytest.raises(ValueError) as excinfo:
            LLMStrategyAnalyzer(provider=provider)
        assert env_var in str(excinfo.value)

def test_llm_strategy_analyzer_invalid_provider():
    with pytest.raises(ValueError) as excinfo:
        LLMStrategyAnalyzer(provider="invalidprovider")
    assert "Unsupported provider" in str(excinfo.value)
