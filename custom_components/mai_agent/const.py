from __future__ import annotations

DOMAIN = "mai_agent"
CONF_API_KEY = "api_key"
CONF_BASE_URL = "base_url"  # OpenAI-compatible endpoint base (e.g. DigitalOcean)
CONF_MODEL = "model"
CONF_MAX_HISTORY = "max_history"
CONF_VERIFY_SSL = "verify_ssl"
CONF_TIMEOUT = "timeout"
CONF_SYSTEM_PROMPT = "system_prompt"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_HISTORY = 8
DEFAULT_VERIFY_SSL = True
DEFAULT_TIMEOUT = 20
DEFAULT_SYSTEM_PROMPT = (
	"Jesteś asystentem sterującym Home Assistant. Masz na imię Jarvis. Odpowiadasz po polsku i generujesz zwięzłe odpowiedzi. Jesteś miły. Gdy zwracasz się po imieniu do użytkownika odnosisz się per pan/pani. "
)

# Dopuszczone domeny i mapowanie prostych poleceń
SUPPORTED_DOMAINS = {"light", "switch", "climate", "fan", "air_purifier", "media_player"}
