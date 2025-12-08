"""Support for EzvizSwitch."""
from __future__ import annotations

import logging
from typing import Any, Dict
from datetime import datetime, timedelta
import voluptuous as vol

try:
    from homeassistant.components.switch import SwitchEntity
except ImportError:
    from homeassistant.components.switch import SwitchDevice as SwitchEntity
from homeassistant.components.switch import PLATFORM_SCHEMA
from homeassistant.const import CONF_EMAIL, CONF_PASSWORD
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from pyezviz import client
from pyezviz.exceptions import (
    EzvizAuthVerificationCode,
    InvalidHost,
    InvalidURL,
    HTTPError,
    PyEzvizError,
)

from pyezviz.constants import (DeviceSwitchType)
from .const import DOMAIN
from .coordinator import EzvizDataUpdateCoordinator

SCAN_INTERVAL = timedelta(seconds=5)
_LOGGER = logging.getLogger(__name__)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_EMAIL): cv.string,
        vol.Optional(CONF_PASSWORD): cv.string,
    }
)


async def async_unload_entry(hass, config_entry):
    """Handle unloading of an entry."""
    _LOGGER.debug(f"async_unload_entry {DOMAIN}: {config_entry}")
    return True


async def async_setup_platform(hass, config, add_entities, discovery_info=None):
    """Perform the setup for Ezviz Smart Plug devices."""

    _LOGGER.debug('calling setup_platform')

    email = config.get(CONF_EMAIL)
    password = config.get(CONF_PASSWORD)
    ezvizClient = client.EzvizClient(email, password)

    try:
        auth_data = await hass.async_add_executor_job(ezvizClient.login)
    except (InvalidHost, InvalidURL, HTTPError, PyEzvizError) as error:
        _LOGGER.exception('Invalid response from API: %s', error)
    except EzvizAuthVerificationCode:
        _LOGGER.exception('MFA Required')
    except (Exception) as error:
        _LOGGER.exception('Unexpected exception: %s', error)

    coordinator = EzvizDataUpdateCoordinator(hass, api=ezvizClient, api_timeout=10)

    # Add devices
    plugs = []
    switches = await coordinator._async_update_data();
    for key, switch in switches.items():
        plugs.append(Ezvizswitch(switch, ezvizClient, coordinator))

    add_entities(plugs)

    _LOGGER.info('Closing the Client session.')
    # Only close if we are done, but here we keep running? 
    # With YAML setup, lifecycle is unclear, but logic matches original.
    # ezvizClient.close_session() 


async def async_setup_entry(hass: core.HomeAssistant, entry: ConfigEntry,
                            async_add_entities: AddEntitiesCallback) -> None:
    """Set up Ezviz switch based on a config entry."""
    entry_data = hass.data[DOMAIN][entry.entry_id]
    coordinator = entry_data["coordinator"]
    ezviz_client = coordinator.ezviz_client

    # Add devices
    plugs = []
    switches = coordinator.data
    for key, switch in switches.items():
        plugs.append(Ezvizswitch(switch, ezviz_client, coordinator))

    async_add_entities(plugs)


class Ezvizswitch(CoordinatorEntity, SwitchEntity, RestoreEntity):
    """Representation of Ezviz Smart Plug Entity."""

    def __init__(self, switch, ezvizClient, coordinator) -> None:
        """Initialize the Ezviz Smart Plug."""
        super().__init__(coordinator)
        self._switch = switch
        self._ezviz_client = ezvizClient
        # self.coordinator is now set provided by CoordinatorEntity

    async def async_added_to_hass(self):
        """Run when entity about to be added."""
        _LOGGER.info('async_added_to_hass called')
        await super().async_added_to_hass()
        # CoordinatorEntity handles update registration

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        if self._switch["deviceSerial"] in self.coordinator.data:
            self._switch = self.coordinator.data[self._switch["deviceSerial"]]
            self.async_write_ha_state()

    async def async_turn_on(self, **kwargs) -> None:
        """Turn device on."""
        _LOGGER.debug('Turning on %s', self._switch['name'])

        # 14 = DeviceSwitchType.PLUG
        result = await self.hass.async_add_executor_job(
            self._ezviz_client.switch_status, self._switch["deviceSerial"], 14, 1
        )
        if result:
            self._switch['enable'] = True
            self.async_write_ha_state()  # Optimistic update
            # Schedule refresh
            await self.coordinator.async_request_refresh()
        else:
            _LOGGER.error("Failed to turn on %s", self.name)

    async def async_turn_off(self, **kwargs) -> None:
        """Turn device off."""
        _LOGGER.debug('Turning off %s', self._switch['name'])

        result = await self.hass.async_add_executor_job(
            self._ezviz_client.switch_status, self._switch["deviceSerial"], 14, 0
        )
        if result:
            self._switch['enable'] = False
            self.async_write_ha_state() # Optimistic update
            # Schedule refresh
            await self.coordinator.async_request_refresh()
        else:
             _LOGGER.error("Failed to turn off %s", self.name)

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        return bool(self._switch.get('enable'))

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._switch.get('status') != 2

    @property
    def unique_id(self) -> str:
        """Return a unique, Home Assistant friendly identifier for this entity."""
        return self._switch['deviceSerial']

    @property
    def name(self) -> str:
        """Return the name of the switch."""
        return self._switch['name']

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        # Removed complexity for now
        return {}

    @property
    def icon(self) -> str:
        """Icon of the entity."""
        if self._switch.get("deviceType", "").endswith("EU"):
            return "mdi:power-socket-de"
        elif self._switch["deviceSerial"].endswith("US"):
            return "mdi:power-socket-us"
        else:
            return "mdi:power-socket"

    @property
    def device_info(self):
        """Return device info to link with the sensor."""
        return {
            "identifiers": {(DOMAIN, self._switch["deviceSerial"])},
            "name": self._switch["name"],
            "manufacturer": "Ezviz",
            "model": self._switch.get("deviceType"),
            "sw_version": self._switch.get("version"),
        }
