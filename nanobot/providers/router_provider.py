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

        if default_model.startswith("hint:"):
            hint = default_model[5:].strip()
            if hint in self._routes:
                self._default_index = self._routes[hint][0]
            else:
                logger.warning("Default model references unknown hint '{}', using first provider", hint)
                self._default_index = 0
        else:
            self._default_index = 0

    def _resolve(self, model: str) -> tuple[int, str, dict[str, Any]]:
        if model.startswith("hint:"):
            hint = model[5:].strip()
            if hint in self._routes:
                idx, resolved_model = self._routes[hint]
                overrides = self._route_overrides.get(hint, {})
                return idx, resolved_model, overrides
            logger.warning("Unknown route hint '{}', falling back to default", hint)
        return self._default_index, model, {}

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
