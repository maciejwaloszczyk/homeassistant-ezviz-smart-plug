"""Provides the ezviz DataUpdateCoordinator."""
from datetime import timedelta
import logging

from async_timeout import timeout
from pyezviz.client import EzvizClient
from pyezviz.exceptions import HTTPError, InvalidURL, PyEzvizError

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class EzvizDataUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage fetching Ezviz data."""

    def __init__(self, hass: HomeAssistant, *, api: EzvizClient, api_timeout: int) -> None:
        """Initialize global Ezviz data updater."""
        self.ezviz_client = api
        self._api_timeout = api_timeout
        update_interval = timedelta(seconds=15)

        super().__init__(hass, _LOGGER, name=DOMAIN, update_interval=update_interval)

    def _update_data(self) -> dict:
        """Fetch Ezviz smart plugs data."""

        plugs = {}
        switches = self.ezviz_client._api_get_pagelist(page_filter="SWITCH")
        for device in switches['deviceInfos']:
            is_plug = False
            for entity in switches['SWITCH'][device['deviceSerial']]:
                if int(entity['type']) is int(14):
                    device['enable'] = entity['enable']
                    is_plug = True

            if is_plug:
                try:
                    # Fetch power data
                    url = f"https://{self.ezviz_client._token['api_url']}/v3/smarthome/outlet/v1/info/op"
                    res = self.ezviz_client._session.get(
                        url,
                        params={"deviceSerial": device['deviceSerial']},
                        timeout=5
                    )
                    res.raise_for_status()
                    power_json = res.json()
                    if power_json.get("meta", {}).get("code") == 200:
                        device["power"] = power_json.get("power")
                except Exception as error:
                    _LOGGER.warning("Could not fetch power for %s: %s", device['deviceSerial'], error)

                plugs[device['deviceSerial']] = device

        return plugs

    async def _async_update_data(self) -> dict:
        """Fetch data from Ezviz."""
        try:
            async with timeout(self._api_timeout):
                return await self.hass.async_add_executor_job(self._update_data)

        except (InvalidURL, HTTPError, PyEzvizError) as error:
            raise UpdateFailed(f"Invalid response from API: {error}") from error