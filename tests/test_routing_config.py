from nanobot.config.schema import Config, ModelEndpoint, RoutingConfig, RoutingCondition, RoutingRule


def test_empty_routing_backward_compatible():
    config = Config()
    assert config.routing.models == {}
    assert config.routing.rules == []


def test_routing_models_parse():
    data = {
        "routing": {
            "models": {
                "default": {"provider": "openrouter", "model": "claude-sonnet-4"},
                "fast": {"provider": "groq", "model": "llama-70b"},
            },
            "rules": [],
        }
    }
    config = Config(**data)
    assert len(config.routing.models) == 2
    assert config.routing.models["default"].provider == "openrouter"
    assert config.routing.models["default"].model == "claude-sonnet-4"
    assert config.routing.models["fast"].provider == "groq"


def test_routing_rules_parse_with_camel_case():
    data = {
        "routing": {
            "models": {
                "default": {"provider": "openrouter", "model": "claude-sonnet-4"},
                "reasoning": {"provider": "openrouter", "model": "claude-opus-4"},
            },
            "rules": [
                {
                    "use": "reasoning",
                    "when": {
                        "keywords": ["explain"],
                        "minLength": 20,
                    },
                },
            ],
        }
    }
    config = Config(**data)
    assert len(config.routing.rules) == 1
    rule = config.routing.rules[0]
    assert rule.use == "reasoning"
    assert rule.when.keywords == ["explain"]
    assert rule.when.min_length == 20
    assert rule.when.max_length is None


def test_model_endpoint_overrides():
    data = {
        "routing": {
            "models": {
                "default": {
                    "provider": "openrouter",
                    "model": "claude-sonnet-4",
                    "maxTokens": 2048,
                    "temperature": 0.5,
                    "reasoningEffort": "low",
                },
            },
        }
    }
    config = Config(**data)
    endpoint = config.routing.models["default"]
    assert endpoint.max_tokens == 2048
    assert endpoint.temperature == 0.5
    assert endpoint.reasoning_effort == "low"


def test_model_endpoint_overrides_default_none():
    data = {
        "routing": {
            "models": {
                "default": {"provider": "groq", "model": "llama-70b"},
            },
        }
    }
    config = Config(**data)
    endpoint = config.routing.models["default"]
    assert endpoint.max_tokens is None
    assert endpoint.temperature is None
    assert endpoint.reasoning_effort is None
