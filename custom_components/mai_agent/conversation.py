"""Conversation platform for Matthias AI Agent (OpenAI-compatible).

Rejestruje encję ConversationEntity + implementuje AbstractConversationAgent.
Używa endpointu OpenAI-kompatybilnego (np. DigitalOcean) do interpretacji języka naturalnego
oraz wywołuje usługi Home Assistant.
"""
from __future__ import annotations

import asyncio
import json
import re
import logging
from typing import Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.components.conversation.const import ConversationEntityFeature
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers import config_validation as cv
import voluptuous as vol

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
    SUPPORTED_DOMAINS,
)

_LOGGER = logging.getLogger(__name__)

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_API_KEY): cv.string,
        vol.Required(CONF_BASE_URL): cv.string,  # Use full base or full chat url
        vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): cv.string,
    vol.Optional(CONF_MAX_HISTORY, default=DEFAULT_MAX_HISTORY): vol.Coerce(int),
    vol.Optional(CONF_VERIFY_SSL, default=DEFAULT_VERIFY_SSL): cv.boolean,
    vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.Coerce(int),
    vol.Optional(CONF_SYSTEM_PROMPT, default=DEFAULT_SYSTEM_PROMPT): cv.string,
    }
)


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    # Setup at component level is no-op; platform handles entities.
    return True


async def async_setup_platform(hass: HomeAssistant, config, async_add_entities: AddEntitiesCallback, discovery_info=None):
    """Legacy setup via configuration.yaml conversation: platform: mai_agent.

    Example:
    conversation:
      - platform: mai_agent
        api_key: !secret do_openai_key
        base_url: https://api.digitalocean.com/v1/ai/openai
        model: gpt-4o-mini
        max_history: 8
    """
    api_key = config.get(CONF_API_KEY)
    base_url = config.get(CONF_BASE_URL)
    model = config.get(CONF_MODEL, DEFAULT_MODEL)
    max_history = int(config.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))
    verify_ssl = bool(config.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL))
    timeout = int(config.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
    system_prompt = str(config.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT))

    if not api_key or not base_url:
        _LOGGER.error("Missing api_key or base_url in mai_agent config")
        return

    yaml_unique = f"yaml_{model}_{abs(hash(base_url))}"
    async_add_entities([HAConversationEntity(api_key, base_url, model, max_history, verify_ssl, timeout, system_prompt, unique_id=yaml_unique)])


async def async_setup_entry(hass: HomeAssistant, entry, async_add_entities: AddEntitiesCallback) -> None:
    """Set up entity from config entry (UI)."""
    data = entry.data
    api_key = data.get(CONF_API_KEY)
    base_url = data.get(CONF_BASE_URL)
    model = data.get(CONF_MODEL, DEFAULT_MODEL)
    max_history = int(data.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))
    verify_ssl = bool(data.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL))
    timeout = int(data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
    # Prefer options-system-prompt over data; fallback to default
    system_prompt = entry.options.get(CONF_SYSTEM_PROMPT, data.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT))

    if not api_key or not base_url:
        _LOGGER.error("Missing api_key or base_url in mai_agent entry")
        return

    async_add_entities([HAConversationEntity(api_key, base_url, model, max_history, verify_ssl, timeout, system_prompt, unique_id=entry.entry_id)])


class HAConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    _attr_supports_streaming = False
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(self, api_key: str, base_url: str, model: str, max_history: int, verify_ssl: bool, timeout: int, system_prompt: str, unique_id: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_history = max_history
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._system_prompt = system_prompt
        self._session: aiohttp.ClientSession | None = None
        self._name = "Matthias AI Agent"
        self._fuzzy_cache: dict[str, str] = {}
        if unique_id:
            self._attr_unique_id = unique_id

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    # For platform-based ConversationEntity we do not set/unset via conversation.async_set_agent.
    # The entity component manages agents automatically.

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _call_llm(self, prompt: str, chat_log: conversation.ChatLog) -> str:
        session = await self._ensure_session()
        # Normalize base URL (add https:// if missing) and construct chat completions path.
        base = self._base_url
        if not base.lower().startswith(("http://", "https://")):
            base = f"https://{base}"
        if base.endswith("/chat/completions"):
            url = base
        elif base.endswith("/v1"):
            url = f"{base}/chat/completions"
        else:
            url = f"{base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        # Build messages from chat_log.content (limited history)
        messages: list[dict] = []
        # Add our system prompt as first message; chat_log system slot is managed by HA
        messages.append({"role": "system", "content": self._system_prompt})
        # Extract last N user/assistant turns
        history: list[dict] = []
        for item in chat_log.content[-(self._max_history*2):]:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            if role in ("user", "assistant") and isinstance(content, str) and content:
                history.append({"role": role, "content": content})
        messages.extend(history)
        # Append current planning prompt as user content
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.2,
        }

        try:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                ssl=self._verify_ssl,
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    _LOGGER.warning("LLM HTTP %s: %s", resp.status, text)
                    raise RuntimeError(f"LLM error {resp.status}")
                try:
                    data = json.loads(text)
                except Exception as parse_err:
                    # Jeśli provider zwraca plain text lub nietypowy JSON, przekaż surową treść dalej
                    # Czasem provider zwraca już finalny content z fenced markdown
                    _LOGGER.debug("LLM non-JSON or unexpected body, passing through: %s", parse_err)
                    return text or ""
        except asyncio.TimeoutError as err:
            raise RuntimeError("LLM timeout") from err
        except aiohttp.ClientError as err:
            raise RuntimeError("LLM network error") from err

        # OpenAI-compatible response parsing
        try:
            content = data["choices"][0]["message"]["content"]
            return content or ""
        except Exception as err:  # noqa: BLE001
            _LOGGER.exception("Invalid LLM response: %s", err)
            raise RuntimeError("Invalid LLM response")

    async def _async_handle_message(self, user_input: conversation.ConversationInput, chat_log: conversation.ChatLog) -> conversation.ConversationResult:
        # Provide context to LLM including available entities and service schema
        entities_hint = self._build_entities_hint()
        # Ask LLM to produce a structured action plan in JSON
        tool_spec = (
            "Zinterpretuj komendę i zwróć JSON (bez formatowania markdown, bez ```), z polami: action (on/off/toggle/set/volume/play/pause), "
            "domain (light/switch/climate/fan/air_purifier/media_player), targets (lista nazw lub entity_id), "
            "attributes (opcjonalne: brightness:0-255, color:hex/nam, temperature:celcius, hvac_mode, fan_mode, volume_level 0-1), "
            "response (krótki komunikat po polsku). Jeśli to tylko pytanie ogólne, ustaw domain=null i zwróć response."
        )
        plan_prompt = f"Kontekst encji:\n{entities_hint}\n\n{tool_spec}\n\nPolecenie: {user_input.text}"

        try:
            llm_text = await self._call_llm(plan_prompt, chat_log)
        except Exception as err:  # Fallback on any LLM error
            _LOGGER.error("LLM call failed: %s", err)
            result = intent.IntentResponse(language=user_input.language or "")
            result.async_set_speech("Wystąpił problem podczas przetwarzania polecenia.")
            return conversation.ConversationResult(response=result, conversation_id=user_input.conversation_id)

        action = None
        domain = None
        targets: list[str] = []
        attributes: dict = {}
        response_text = None

        # Try parse JSON (also if wrapped in ```json ... ```)
        clean_text = self._strip_code_fences(llm_text)
        parsed = None
        try:
            parsed = json.loads(clean_text)
        except Exception:
            parsed = self._extract_json_object(clean_text)

        if isinstance(parsed, dict):
            action = parsed.get("action")
            domain = parsed.get("domain")
            targets = parsed.get("targets") or []
            attributes = parsed.get("attributes") or {}
            response_text = parsed.get("response")
            # Fallbacky dla niekanonicznych pól od providerów
            if not response_text and isinstance(parsed.get("message"), str):
                response_text = parsed.get("message")
            # Obsługa pojedynczych pól celu
            if not targets:
                if isinstance(parsed.get("entity_id"), str):
                    targets = [parsed.get("entity_id")]
                elif isinstance(parsed.get("entity_id"), list):
                    targets = parsed.get("entity_id")
                elif isinstance(parsed.get("target"), str):
                    targets = [parsed.get("target")]
                elif isinstance(parsed.get("target"), list):
                    targets = parsed.get("target")
            # Mapowanie device -> domain, jeśli brak domain
            if not domain and isinstance(parsed.get("device"), str):
                dev = parsed.get("device").strip().lower()
                device_to_domain = {
                    "air_purifier": "fan",  # większość oczyszczaczy to domena fan
                    "oczyszczacz": "fan",
                    "tv": "media_player",
                    "telewizor": "media_player",
                    "speaker": "media_player",
                    "głośnik": "media_player",
                    "ac": "climate",
                    "klimatyzacja": "climate",
                    "light": "light",
                    "lampa": "light",
                    "światło": "light",
                    "switch": "switch",
                }
                domain = device_to_domain.get(dev, domain)
            # Porządkuj action
            if isinstance(action, str):
                action = action.strip()
        else:
            # If model answered text, just return as response
            response_text = clean_text.strip()

        # Handle domain-specific actions or fallback to plain response
        if domain and domain not in SUPPORTED_DOMAINS:
            response_text = response_text or "Nie mogę wykonać tej akcji."
            result = intent.IntentResponse(language=user_input.language or "")
            result.async_set_speech(response_text)
            return conversation.ConversationResult(response=result, conversation_id=user_input.conversation_id)

        if domain:
            # Resolve targets -> entity_ids via fuzzy match or direct id
            entity_ids = self._resolve_targets(targets, domain)
            if not entity_ids:
                # Ask a follow-up
                # Jeśli brak targetów i istnieje dokładnie jedna encja w tej domenie, użyj jej automatycznie
                if not targets:
                    candidates = [st.entity_id for st in self.hass.states.async_all() if st.domain == domain]
                    if len(candidates) == 1:
                        entity_ids = candidates
                    else:
                        ask = "Które urządzenie masz na myśli? Podaj nazwę lub pomieszczenie."
                        result = intent.IntentResponse(language=user_input.language or "")
                        result.async_set_speech(ask)
                        return conversation.ConversationResult(response=result, conversation_id=user_input.conversation_id, continue_conversation=True)

            # Map action to service and data
            service, data = self._map_action_to_service(domain, action, attributes)
            if not service:
                result = intent.IntentResponse(language=user_input.language or "")
                result.async_set_speech("Nie rozumiem działania.")
                return conversation.ConversationResult(response=result, conversation_id=user_input.conversation_id)

            for eid in entity_ids:
                call = {"entity_id": eid}
                call.update(data)
                await self.hass.services.async_call(domain, service, call, blocking=False)

            spoken = response_text or self._make_confirmation(domain, action, entity_ids, attributes)
            result = intent.IntentResponse(language=user_input.language or "")
            result.async_set_speech(spoken)
            return conversation.ConversationResult(response=result, conversation_id=user_input.conversation_id)

        # No domain -> just answer
        if not response_text:
            response_text = "Cześć! W czym mogę pomóc?"
        result = intent.IntentResponse(language=user_input.language or "")
        result.async_set_speech(response_text)
        return conversation.ConversationResult(response=result, conversation_id=user_input.conversation_id)

    def _strip_code_fences(self, text: str) -> str:
        s = text.strip()
        if s.startswith("```"):
            # Drop first line (``` or ```json)
            nl = s.find("\n")
            if nl != -1:
                s = s[nl + 1 :]
            # Remove trailing ``` if present
            if s.endswith("```"):
                s = s[:-3]
        return s.strip()

    def _extract_json_object(self, text: str) -> dict | None:
        # Find a JSON object substring
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            return None

    def _build_entities_hint(self) -> str:
        states = self.hass.states.async_all()
        lines: list[str] = []
        for st in states:
            domain = st.domain
            if domain not in SUPPORTED_DOMAINS:
                continue
            name = st.name or st.entity_id
            lines.append(f"{domain}:{st.entity_id}:{name}")
            self._fuzzy_cache[name.lower()] = st.entity_id
        return "\n".join(lines[:200])

    def _resolve_targets(self, targets: list[str], domain: str) -> list[str]:
        result: list[str] = []
        for t in targets:
            t = str(t).strip()
            if not t:
                continue
            if "." in t:
                # Looks like entity_id
                if t.split(".",1)[0] == domain:
                    result.append(t)
                    continue
            key = t.lower()
            if key in self._fuzzy_cache:
                eid = self._fuzzy_cache[key]
                if eid.split(".",1)[0] == domain:
                    result.append(eid)
        return result

    def _map_action_to_service(self, domain: str, action: str | None, attrs: dict):
        if not action:
            return None, {}
        action = action.lower()
        if action in ("on", "turn_on"):
            return "turn_on", {}
        if action in ("off", "turn_off"):
            return "turn_off", {}
        if action in ("toggle",):
            return "toggle", {}
        if domain == "light" and action == "set":
            data = {}
            if "brightness" in attrs:
                try:
                    b = int(attrs["brightness"])  # 0-255
                    data["brightness"] = max(0, min(255, b))
                except Exception:
                    pass
            # color or temperature could be mapped here if available
            return "turn_on", data
        if domain == "climate" and action == "set":
            data = {}
            if "temperature" in attrs:
                try:
                    data["temperature"] = float(attrs["temperature"])
                except Exception:
                    pass
            if "hvac_mode" in attrs:
                data["hvac_mode"] = str(attrs["hvac_mode"]).lower()
            if data:
                return "set_temperature", data
            return None, {}
        if domain in ("fan", "air_purifier") and action in ("on", "off"):
            return ("turn_on" if action == "on" else "turn_off"), {}
        if domain == "media_player":
            if action == "play":
                return "media_play", {}
            if action == "pause":
                return "media_pause", {}
            if action == "stop":
                return "media_stop", {}
            if action == "volume":
                data = {}
                if "volume_level" in attrs:
                    try:
                        v = float(attrs["volume_level"])
                        data["volume_level"] = max(0.0, min(1.0, v))
                    except Exception:
                        pass
                if data:
                    return "volume_set", data
                return None, {}
        return None, {}

    def _make_confirmation(self, domain: str, action: str | None, eids: list[str], attrs: dict) -> str:
        pretty = ", ".join(eids[:3])
        if action in ("on", "turn_on"):
            return f"Włączam: {pretty}."
        if action in ("off", "turn_off"):
            return f"Wyłączam: {pretty}."
        if action == "toggle":
            return f"Przełączam: {pretty}."
        if domain == "light" and action == "set":
            if "brightness" in attrs:
                return f"Ustawiam jasność na {attrs['brightness']}."
            return f"Aktualizuję światła: {pretty}."
        if domain == "climate" and action == "set":
            return "Ustawiam klimatyzację."
        if domain == "media_player":
            if action == "play":
                return "Odtwarzam."
            if action == "pause":
                return "Pauzuję."
            if action == "stop":
                return "Zatrzymuję."
            if action == "volume":
                return "Ustawiam głośność."
        return "Wykonane."
