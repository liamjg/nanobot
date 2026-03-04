from nanobot.agent.classifier import classify
from nanobot.config.schema import ModelEndpoint, RoutingCondition, RoutingConfig, RoutingRule


def _routing(rules=None, models=None):
    if models is None:
        models = {
            "default": ModelEndpoint(provider="test", model="default-model"),
            "fast": ModelEndpoint(provider="test", model="fast-model"),
            "reasoning": ModelEndpoint(provider="test", model="reasoning-model"),
        }
    return RoutingConfig(models=models, rules=rules or [])


def test_no_rules_returns_none():
    routing = _routing(rules=[])
    assert classify(routing, "hello") is None


def test_keyword_match_case_insensitive():
    routing = _routing(rules=[
        RoutingRule(use="fast", when=RoutingCondition(keywords=["hello"])),
    ])
    assert classify(routing, "HELLO world") == "fast"


def test_pattern_match_case_sensitive():
    routing = _routing(rules=[
        RoutingRule(use="reasoning", when=RoutingCondition(patterns=["fn "])),
    ])
    assert classify(routing, "fn main()") == "reasoning"
    assert classify(routing, "FN MAIN()") is None


def test_min_length_constraint():
    routing = _routing(rules=[
        RoutingRule(use="reasoning", when=RoutingCondition(keywords=["explain"], min_length=20)),
    ])
    assert classify(routing, "explain") is None
    assert classify(routing, "explain how this works in detail") == "reasoning"


def test_max_length_constraint():
    routing = _routing(rules=[
        RoutingRule(use="fast", when=RoutingCondition(keywords=["hi"], max_length=10)),
    ])
    assert classify(routing, "hi") == "fast"
    assert classify(routing, "hi there, how are you doing today?") is None


def test_list_order_determines_precedence():
    routing = _routing(rules=[
        RoutingRule(use="reasoning", when=RoutingCondition(keywords=["code"])),
        RoutingRule(use="fast", when=RoutingCondition(keywords=["code"])),
    ])
    assert classify(routing, "write some code") == "reasoning"


def test_rule_referencing_undefined_model_skipped():
    routing = _routing(
        models={"default": ModelEndpoint(provider="test", model="m")},
        rules=[
            RoutingRule(use="nonexistent", when=RoutingCondition(keywords=["hello"])),
            RoutingRule(use="default", when=RoutingCondition(keywords=["hello"])),
        ],
    )
    assert classify(routing, "hello") == "default"


def test_no_match_returns_none():
    routing = _routing(rules=[
        RoutingRule(use="fast", when=RoutingCondition(keywords=["hello"])),
    ])
    assert classify(routing, "something completely different") is None
