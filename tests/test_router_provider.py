import pytest

from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.router_provider import RouterProvider


class MockProvider(LLMProvider):
    def __init__(self, name: str, default_model: str = "mock-model"):
        super().__init__(api_key="test")
        self._name = name
        self._default_model = default_model
        self.calls: list[dict] = []

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append({"model": model, **kwargs})
        return LLMResponse(content=f"response-from-{self._name}")

    def get_default_model(self):
        return self._default_model


def _make_router(routes=None, overrides=None):
    fast = MockProvider("fast")
    smart = MockProvider("smart")
    providers = [("fast", fast), ("smart", smart)]
    route_table = routes or {"reasoning": (1, "claude-opus"), "fast": (0, "llama-70b")}
    return RouterProvider(
        providers=providers,
        routes=route_table,
        default_model="default-model",
        route_overrides=overrides,
    ), fast, smart


@pytest.mark.asyncio
async def test_hint_routes_to_correct_provider():
    router, fast, smart = _make_router()
    result = await router.chat([], model="hint:reasoning")
    assert result.content == "response-from-smart"
    assert len(smart.calls) == 1
    assert smart.calls[0]["model"] == "claude-opus"
    assert len(fast.calls) == 0


@pytest.mark.asyncio
async def test_hint_routes_fast():
    router, fast, smart = _make_router()
    result = await router.chat([], model="hint:fast")
    assert result.content == "response-from-fast"
    assert fast.calls[0]["model"] == "llama-70b"


@pytest.mark.asyncio
async def test_unknown_hint_falls_back_to_default():
    router, fast, smart = _make_router()
    result = await router.chat([], model="hint:nonexistent")
    assert result.content == "response-from-fast"
    assert fast.calls[0]["model"] == "hint:nonexistent"


@pytest.mark.asyncio
async def test_non_hint_model_uses_default():
    router, fast, smart = _make_router()
    result = await router.chat([], model="anthropic/claude-sonnet-4")
    assert result.content == "response-from-fast"
    assert fast.calls[0]["model"] == "anthropic/claude-sonnet-4"


@pytest.mark.asyncio
async def test_route_overrides_applied():
    overrides = {"reasoning": {"max_tokens": 16384, "temperature": 0.3}}
    router, fast, smart = _make_router(overrides=overrides)
    await router.chat([], model="hint:reasoning")
    assert smart.calls[0]["max_tokens"] == 16384
    assert smart.calls[0]["temperature"] == 0.3


@pytest.mark.asyncio
async def test_route_without_overrides_passes_no_extra_params():
    overrides = {"reasoning": {"max_tokens": 16384}}
    router, fast, smart = _make_router(overrides=overrides)
    await router.chat([], model="hint:fast")
    assert "max_tokens" not in fast.calls[0]
    assert "temperature" not in fast.calls[0]


@pytest.mark.asyncio
async def test_new_sampling_params_forwarded():
    overrides = {"reasoning": {
        "top_p": 0.9, "top_k": 40, "min_p": 0.05,
        "frequency_penalty": 0.5, "presence_penalty": 0.3,
    }}
    router, fast, smart = _make_router(overrides=overrides)
    await router.chat([], model="hint:reasoning")
    assert smart.calls[0]["top_p"] == 0.9
    assert smart.calls[0]["top_k"] == 40
    assert smart.calls[0]["min_p"] == 0.05
    assert smart.calls[0]["frequency_penalty"] == 0.5
    assert smart.calls[0]["presence_penalty"] == 0.3


def test_get_default_model():
    router, _, _ = _make_router()
    assert router.get_default_model() == "default-model"


@pytest.mark.asyncio
async def test_none_model_uses_default():
    router, fast, _ = _make_router()
    await router.chat([], model=None)
    assert fast.calls[0]["model"] == "default-model"


@pytest.mark.asyncio
async def test_hint_default_model_resolves_to_correct_provider():
    fast = MockProvider("fast")
    smart = MockProvider("smart")
    providers = [("fast", fast), ("smart", smart)]
    routes = {"reasoning": (1, "claude-opus"), "fast": (0, "llama-70b")}
    router = RouterProvider(
        providers=providers,
        routes=routes,
        default_model="hint:reasoning",
    )
    result = await router.chat([], model=None)
    assert result.content == "response-from-smart"
    assert smart.calls[0]["model"] == "claude-opus"
    assert len(fast.calls) == 0


@pytest.mark.asyncio
async def test_hint_default_model_unknown_hint_falls_back_to_first():
    fast = MockProvider("fast")
    smart = MockProvider("smart")
    providers = [("fast", fast), ("smart", smart)]
    routes = {"reasoning": (1, "claude-opus")}
    router = RouterProvider(
        providers=providers,
        routes=routes,
        default_model="hint:nonexistent",
    )
    result = await router.chat([], model=None)
    assert result.content == "response-from-fast"


@pytest.mark.asyncio
async def test_hint_default_still_allows_explicit_hint():
    fast = MockProvider("fast")
    smart = MockProvider("smart")
    providers = [("fast", fast), ("smart", smart)]
    routes = {"reasoning": (1, "claude-opus"), "fast": (0, "llama-70b")}
    router = RouterProvider(
        providers=providers,
        routes=routes,
        default_model="hint:reasoning",
    )
    result = await router.chat([], model="hint:fast")
    assert result.content == "response-from-fast"
    assert fast.calls[0]["model"] == "llama-70b"
