from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_MAX_HISTORY,
    CONF_VERIFY_SSL,
    CONF_TIMEOUT,
    CONF_SYSTEM_PROMPT,
    DEFAULT_MODEL,
    DEFAULT_MAX_HISTORY,
    DEFAULT_VERIFY_SSL,
    DEFAULT_TIMEOUT,
    DEFAULT_SYSTEM_PROMPT,
)


class HAAgentConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            # Minimal validation
            if not user_input.get(CONF_API_KEY) or not user_input.get(CONF_BASE_URL):
                errors["base"] = "missing_required"
            else:
                return self.async_create_entry(title="Matthias AI Agent", data=user_input)

        data_schema = vol.Schema(
            {
                vol.Required(CONF_API_KEY): str,
                vol.Required(CONF_BASE_URL): str,
                vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): str,
                vol.Optional(CONF_MAX_HISTORY, default=DEFAULT_MAX_HISTORY): int,
                vol.Optional(CONF_VERIFY_SSL, default=DEFAULT_VERIFY_SSL): bool,
                vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): int,
            }
        )

        return self.async_show_form(step_id="user", data_schema=data_schema, errors=errors)


class HAAgentOptionsFlowHandler(config_entries.OptionsFlow):
    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            # Zapisz tylko opcje; nie dotykamy data
            return self.async_create_entry(title="", data=user_input)

        # Domyślna wartość: options -> data -> DEFAULT
        current = self.config_entry.options.get(
            CONF_SYSTEM_PROMPT,
            self.config_entry.data.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
        )
        data_schema = vol.Schema({
            vol.Optional(
                CONF_SYSTEM_PROMPT,
                description={"suggested_value": current},
            ): selector.TextSelector(selector.TextSelectorConfig(multiline=True))
        })
        return self.async_show_form(step_id="init", data_schema=data_schema)


async def async_get_options_flow(config_entry: config_entries.ConfigEntry) -> HAAgentOptionsFlowHandler:
    return HAAgentOptionsFlowHandler(config_entry)
