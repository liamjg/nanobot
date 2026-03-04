from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse


class RouterProvider(LLMProvider):

    def __init__(
        self,
        providers: list[tuple[str, LLMProvider]],
        routes: dict[str, tuple[int, str]],
        default_model: str,
        route_overrides: dict[str, dict[str, Any]] | None = None,
    ):
        super().__init__(api_key=None, api_base=None)
        self._providers = providers
        self._routes = routes
        self._default_model = default_model
        self._route_overrides = route_overrides or {}

        if default_model not in self._routes:
            raise ValueError(f"Default model '{default_model}' not found in routes: {list(self._routes)}")

    def _resolve(self, model: str) -> tuple[int, str, dict[str, Any]]:
        if model not in self._routes:
            raise ValueError(f"Unknown route '{model}', valid routes: {list(self._routes)}")
        idx, resolved_model = self._routes[model]
        overrides = self._route_overrides.get(model, {})
        return idx, resolved_model, overrides

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        effective_model = model or self._default_model
        provider_idx, resolved_model, overrides = self._resolve(effective_model)
        name, provider = self._providers[provider_idx]

        logger.info("Router dispatching to provider='{}', model='{}'", name, resolved_model)

        return await provider.chat(
            messages=messages,
            tools=tools,
            model=resolved_model,
            **overrides,
        )

    def get_default_model(self) -> str:
        return self._default_model
