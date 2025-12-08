"""Support for Ezviz Power Sensors."""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict

import voluptuous as vol

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
    PLATFORM_SCHEMA,
)
from homeassistant.const import CONF_EMAIL, CONF_PASSWORD, UnitOfPower
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddEntitiesCallback
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.restore_state import RestoreEntity

from pyezviz import client
from pyezviz.exceptions import (
    EzvizAuthVerificationCode,
    InvalidHost,
    InvalidURL,
    HTTPError,
    PyEzvizError,
)

from .const import DOMAIN
from .coordinator import EzvizDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_EMAIL): cv.string,
        vol.Optional(CONF_PASSWORD): cv.string,
    }
)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Ezviz sensor based on a config entry."""
    entry_data = hass.data[DOMAIN][entry.entry_id]
    coordinator = entry_data["coordinator"]
    ezviz_client = coordinator.ezviz_client

    # Fetch initial data
    plugs_data = coordinator.data
    sensors = []
    for serial, device_data in plugs_data.items():
        if "power" in device_data:
            sensors.append(EzvizPowerSensor(device_data, ezviz_client, coordinator))

    async_add_entities(sensors)


from homeassistant.core import callback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

class EzvizPowerSensor(CoordinatorEntity, SensorEntity, RestoreEntity):
    """Representation of an Ezviz Power Sensor."""

    def __init__(self, device_data, ezvizClient, coordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._device_data = device_data
        self._ezviz_client = ezvizClient
        # self.coordinator provided by super
        self._attr_name = f"{device_data['name']} Power"
        self._attr_unique_id = f"{device_data['deviceSerial']}_power"
        self._attr_device_class = SensorDeviceClass.POWER
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_native_unit_of_measurement = UnitOfPower.WATT

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        if self._device_data['deviceSerial'] in self.coordinator.data:
            self._device_data = self.coordinator.data[self._device_data['deviceSerial']]
            self.async_write_ha_state()

    @property
    def native_value(self) -> float | None:
        """Return the state of the sensor."""
        # Use latest data from coordinator
        val = self._device_data.get("power")
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    @property
    def icon(self) -> str:
        """Icon of the entity."""
        return "mdi:flash"
        
    @property
    def device_info(self):
        """Return device info to link with the switch."""
        return {
            "identifiers": {(DOMAIN, self._device_data["deviceSerial"])},
            "name": self._device_data["name"],
            "manufacturer": "Ezviz",
            "model": self._device_data.get("deviceType"),
            "sw_version": self._device_data.get("version"),
        }

