"""Ezviz Smart Plug integration."""

from homeassistant import config_entries, core
from homeassistant.const import Platform

from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.const import CONF_EMAIL, CONF_PASSWORD
from pyezviz.client import EzvizClient
from pyezviz.exceptions import (
    EzvizAuthVerificationCode,
    InvalidHost,
    InvalidURL,
    HTTPError,
    PyEzvizError,
)
from .coordinator import EzvizDataUpdateCoordinator
from .const import DOMAIN

PLATFORMS = [Platform.SWITCH, Platform.SENSOR]


async def async_setup_entry(hass: core.HomeAssistant, entry: config_entries.ConfigEntry) -> bool:
    """Set up platform from a ConfigEntry."""

    hass.data.setdefault(DOMAIN, {})
    
    email = entry.data.get(CONF_EMAIL)
    password = entry.data.get(CONF_PASSWORD)

    if not email or not password:
        return False

    ezviz_client = EzvizClient(email, password)
    try:
        await hass.async_add_executor_job(ezviz_client.login)
    except (InvalidHost, InvalidURL, HTTPError, PyEzvizError) as error:
        raise ConfigEntryNotReady(f"Connection failed: {error}") from error
    except EzvizAuthVerificationCode:
        return False
    except Exception as error:
        raise ConfigEntryNotReady(f"Unexpected exception: {error}") from error

    coordinator = EzvizDataUpdateCoordinator(hass, api=ezviz_client, api_timeout=10)
    await coordinator.async_config_entry_first_refresh()

    hass_data = dict(entry.data)
    hass_data["coordinator"] = coordinator

    # Registers update listener to update config entry when options are updated.
    unsub_options_update_listener = entry.add_update_listener(options_update_listener)
    hass_data["unsub_options_update_listener"] = unsub_options_update_listener
    
    hass.data[DOMAIN][entry.entry_id] = hass_data

    # Forward the setup to the platforms.
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: core.HomeAssistant, entry: config_entries.ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        # Remove options_update_listener.
        hass.data[DOMAIN][entry.entry_id]["unsub_options_update_listener"]()
        
        # Remove config entry from domain.
        entry_data = hass.data[DOMAIN].pop(entry.entry_id)
        if "coordinator" in entry_data:
            entry_data["coordinator"].ezviz_client.close_session()
        
    return unload_ok


async def options_update_listener(hass: core.HomeAssistant, config_entry: config_entries.ConfigEntry):
    """Handle options update."""
    await hass.config_entries.async_reload(config_entry.entry_id)


async def async_setup(hass: core.HomeAssistant, config: dict) -> bool:
    """Set up the Ezviz Smart Plug custom component from yaml configuration."""
    hass.data.setdefault(DOMAIN, {})
    return True
