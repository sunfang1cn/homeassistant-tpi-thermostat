from __future__ import annotations

from homeassistant.components.climate import ClimateEntity, PLATFORM_SCHEMA
from homeassistant.const import  ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.components.climate.const import   ClimateEntityFeature, HVACMode, HVACAction
import homeassistant.helpers.config_validation as cv
from homeassistant.components.mqtt.mixins import  MqttEntity
from homeassistant.components.mqtt.schemas import MQTT_ENTITY_COMMON_SCHEMA
from homeassistant.components.mqtt.const import     CONF_COMMAND_TEMPLATE
from homeassistant.components.mqtt.config import MQTT_RW_SCHEMA
from homeassistant.components.input_text import SERVICE_SET_VALUE, ATTR_VALUE, DOMAIN as INPUT_TEXT_DOMAIN
import voluptuous as vol

import asyncio
from datetime import datetime, timedelta
import datetime as dt
import logging
import math, time
from typing import Any
from . import DOMAIN, PLATFORMS
import voluptuous as vol
from homeassistant.components import climate
from homeassistant.components.climate import (
    ATTR_PRESET_MODE,
    PLATFORM_SCHEMA,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
    PRESET_ACTIVITY,
    PRESET_AWAY,
    PRESET_COMFORT,
    PRESET_ECO,
    PRESET_NONE,
    PRESET_SLEEP,
    PRESET_HOME
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_TEMPERATURE,
    CONF_NAME,
    CONF_VALUE_TEMPLATE,
    CONF_UNIQUE_ID,
    EVENT_HOMEASSISTANT_START,
    PRECISION_HALVES,
    PRECISION_TENTHS,
    PRECISION_WHOLE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfTemperature,
)
from homeassistant.core import (
    DOMAIN as HA_DOMAIN,
    CoreState,
    Event,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    EventStateChangedData,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, EventType
from homeassistant.const import Platform

MQTT_CLIMATE_ATTRIBUTES_BLOCKED = frozenset(
    {
        climate.ATTR_AUX_HEAT,
        climate.ATTR_CURRENT_HUMIDITY,
        climate.ATTR_CURRENT_TEMPERATURE,
        climate.ATTR_FAN_MODE,
        climate.ATTR_FAN_MODES,
        climate.ATTR_HUMIDITY,
        climate.ATTR_HVAC_ACTION,
        climate.ATTR_HVAC_MODES,
        climate.ATTR_MAX_HUMIDITY,
        climate.ATTR_MAX_TEMP,
        climate.ATTR_MIN_HUMIDITY,
        climate.ATTR_MIN_TEMP,
        climate.ATTR_PRESET_MODE,
        climate.ATTR_PRESET_MODES,
        climate.ATTR_SWING_MODE,
        climate.ATTR_SWING_MODES,
        climate.ATTR_TARGET_TEMP_HIGH,
        climate.ATTR_TARGET_TEMP_LOW,
        climate.ATTR_TARGET_TEMP_STEP,
        climate.ATTR_TEMPERATURE,
    }
)
DEFAULT_NAME = "MQTT Select"
CONF_OPTIONS = 'OPTIONS'
PLATFORM_SCHEMA_MODERN = MQTT_RW_SCHEMA.extend(
    {
        vol.Optional(CONF_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_OPTIONS): cv.ensure_list,
        vol.Optional(CONF_VALUE_TEMPLATE): cv.template,
    },
).extend(MQTT_ENTITY_COMMON_SCHEMA.schema)
DISCOVERY_SCHEMA = vol.All(PLATFORM_SCHEMA_MODERN.extend({}, extra=vol.REMOVE_EXTRA))


import logging
_LOGGER = logging.getLogger(__name__)


DEFAULT_TOLERANCE = 0.5
DEFAULT_DEADHAND = 2
DEFAULT_NAME = "Tpi Thermostat"
DEFAULT_TPI_KP = 0.606
DEFAULT_TPI_TI = 800.00
DEFAULT_MIN_HEATER_TEMP = 40.0
DEFAULT_MAX_HEATER_TEMP = 70.0

CONF_HEATER_SWITCH = "heater_switch"
CONF_HEATER_SET_TEMP_MQTT_TOPIC = "heater_temp_topic"
CONF_HEATER_SET_TEMP_MQTT_PAYLOAD = "heater_temp_payload"
CONF_DEADHAND = "deadhand"
CONF_SENSOR = "target_sensor"
CONF_MIN_TEMP = "min_temp"
CONF_MAX_TEMP = "max_temp"
CONF_TARGET_TEMP = "target_temp"
CONF_AC_MODE = "ac_mode"
CONF_MIN_DUR = "min_cycle_duration"
CONF_COLD_TOLERANCE = "cold_tolerance"
CONF_HOT_TOLERANCE = "hot_tolerance"
CONF_KEEP_ALIVE = "keep_alive"
CONF_INITIAL_HVAC_MODE = "initial_hvac_mode"
CONF_PRECISION = "precision"
CONF_TEMP_STEP = "target_temp_step"
CONF_TPI_KP = 'tpi_kp'
CONF_TPI_TI = 'tpi_ti'
CONF_AUTO_TPI = 'auto_tpi' 
CONF_MIN_HEATER_TEMP = 'heater_temp_min'
CONF_MAX_HEATER_TEMP = 'heater_temp_max'
DEFALUT_KEEP_ALIVE = dt.timedelta(seconds=60)
SLOPE_TABLE_START_KP = 0.966
SLOPE_TABLE_START_TI = 1100.0
SLOPE_TABLE_START_SLOPE = 10.0
SLOPE_TABLE_MAX_SLOPE = 70.0
SLOPE_TABLE_STEP_KP = 0.01333
SLOPE_TABLE_STEP_TI = 10.0
CONF_TEMP_INPUT_ID = 'tempurature_input_entity'

CONF_PRESETS = {
    p: f"{p}_temp"
    for p in (
        PRESET_AWAY,
        PRESET_COMFORT,
        PRESET_ECO,
        PRESET_HOME,
        PRESET_SLEEP,
        PRESET_ACTIVITY,
    )
}

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_HEATER_SWITCH): cv.entity_id,
        vol.Required(CONF_SENSOR): cv.entity_id,

        vol.Optional(CONF_HEATER_SET_TEMP_MQTT_TOPIC, default="not_set"): cv.string,
        vol.Optional(CONF_HEATER_SET_TEMP_MQTT_PAYLOAD, default="not_set"): cv.string,
        vol.Optional(CONF_TEMP_INPUT_ID, default="not_set"): cv.string,

        vol.Optional(CONF_AC_MODE, default=False): cv.boolean,
        vol.Optional(CONF_MAX_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MIN_DUR): cv.positive_time_period,
        vol.Optional(CONF_MIN_TEMP): vol.Coerce(float),
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_COLD_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
        vol.Optional(CONF_HOT_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
        vol.Optional(CONF_DEADHAND, default=DEFAULT_DEADHAND): vol.Coerce(float),
        vol.Optional(CONF_AUTO_TPI, default=True): cv.boolean,

        vol.Optional(CONF_TPI_KP, default=DEFAULT_TPI_KP): vol.Coerce(float),
        vol.Optional(CONF_TPI_TI, default=DEFAULT_TPI_TI): vol.Coerce(float),
        vol.Optional(CONF_MAX_HEATER_TEMP, default=DEFAULT_MAX_HEATER_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MIN_HEATER_TEMP, default=DEFAULT_MIN_HEATER_TEMP): vol.Coerce(float),
        vol.Optional(CONF_TARGET_TEMP): vol.Coerce(float),
        vol.Optional(CONF_KEEP_ALIVE, default=DEFALUT_KEEP_ALIVE): cv.positive_time_period,
        vol.Optional(CONF_INITIAL_HVAC_MODE): vol.In(
            [ HVACMode.HEAT, HVACMode.OFF]
        ),
        vol.Optional(CONF_PRECISION): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        ),
        vol.Optional(CONF_TEMP_STEP): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        ),
        vol.Optional(CONF_UNIQUE_ID): cv.string,
    }
).extend({vol.Optional(v): vol.Coerce(float) for (k, v) in CONF_PRESETS.items()})


async def async_setup_platform(hass: HomeAssistant, 
                               config: ConfigType,
                               async_add_entities: AddEntitiesCallback, 
                               discovery_info: DiscoveryInfoType  | None=None):
    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)

    name: str = config[CONF_NAME]
    heater_switch_entity_id: str = config[CONF_HEATER_SWITCH]
    heater_temp_topic: str | None = config[CONF_HEATER_SET_TEMP_MQTT_TOPIC]
    heater_temp_payload: str | None = config[CONF_HEATER_SET_TEMP_MQTT_PAYLOAD]
    heater_temp_input_entity_id: str | None = config.get(CONF_TEMP_INPUT_ID)
    sensor_entity_id: str = config[CONF_SENSOR]
    
    if ( heater_temp_topic == 'not_set' or heater_temp_payload == 'not_set' ) and heater_temp_input_entity_id == 'not_set':
        _LOGGER.error("One of heater_temp_topic or heater_temp_input_entity_id must be set!")
        return

    deadhand: float | None = config.get(CONF_DEADHAND)
    auto_tpi: bool | None = config.get(CONF_AUTO_TPI)
    tpi_kp: float | None = config.get(CONF_TPI_KP)
    tpi_ti: float | None = config.get(CONF_TPI_TI)

    min_temp: float | None = config.get(CONF_MIN_TEMP)
    max_temp: float | None = config.get(CONF_MAX_TEMP)
    min_heater_temp: float | None = config.get(CONF_MIN_HEATER_TEMP)
    max_heater_temp: float | None = config.get(CONF_MAX_HEATER_TEMP)

    target_temp: float | None = config.get(CONF_TARGET_TEMP)
    ac_mode: bool | None = config.get(CONF_AC_MODE)
    min_cycle_duration: timedelta | None = config.get(CONF_MIN_DUR)
    cold_tolerance: float = config[CONF_COLD_TOLERANCE]
    hot_tolerance: float = config[CONF_HOT_TOLERANCE]
    keep_alive: timedelta | None = config.get(CONF_KEEP_ALIVE)
    initial_hvac_mode: HVACMode | None = config.get(CONF_INITIAL_HVAC_MODE)
    presets: dict[str, float] = {
        key: config[value] for key, value in CONF_PRESETS.items() if value in config
    }
    precision: float | None = config.get(CONF_PRECISION)
    target_temperature_step: float | None = config.get(CONF_TEMP_STEP)
    unit = hass.config.units.temperature_unit
    unique_id: str | None = config.get(CONF_UNIQUE_ID)

    async_add_entities(
        [
            TpiThermostat(
                name,
                heater_switch_entity_id,
                heater_temp_topic,
                heater_temp_payload,
                deadhand,
                auto_tpi,
                tpi_kp,
                tpi_ti,
                sensor_entity_id,
                min_temp,
                max_temp,
                target_temp,
                ac_mode,
                min_cycle_duration,
                cold_tolerance,
                hot_tolerance,
                keep_alive,
                initial_hvac_mode,
                presets,
                precision,
                target_temperature_step,
                unit,
                unique_id,
                min_heater_temp,
                max_heater_temp,
                hass, config, async_add_entities, discovery_info,
                heater_temp_input_entity_id
            )
        ]
    )


class TpiThermostat(ClimateEntity, RestoreEntity, MqttEntity):
    _attr_should_poll = False
    _enable_turn_on_off_backwards_compatibility = False
    _entity_id_format = climate.ENTITY_ID_FORMAT
    _attributes_extra_blocked = MQTT_CLIMATE_ATTRIBUTES_BLOCKED

    def __init__(self, 
                name,
                heater_switch_entity_id,
                heater_temp_topic,
                heater_temp_payload,
                deadhand,
                auto_tpi,
                tpi_kp,
                tpi_ti,
                sensor_entity_id,
                min_temp,
                max_temp,
                target_temp,
                ac_mode,
                min_cycle_duration,
                cold_tolerance,
                hot_tolerance,
                keep_alive,
                initial_hvac_mode,
                presets,
                precision,
                target_temperature_step,
                unit: UnitOfTemperature,
                unique_id,
                min_heater_temp,
                max_heater_temp,
                hass, conf, async_add_entities, discovery_info,
                heater_temp_input_entity_id
                ):
        

        
        MqttEntity.__init__(self, hass, conf, async_add_entities, discovery_info)

        self.current_state = 0
        self.tpi_error_old = 0
        self.tpi_out_old = 0
        self.tpi_start = 1
        self.tpi_active = False
        self.last_deadhand_reason = None

        """Initialize the thermostat."""
        self._attr_name = name
        self.heater_switch_entity_id = heater_switch_entity_id
        self.heater_temp_topic = heater_temp_topic
        self.heater_temp_payload = heater_temp_payload
        self.deadhand = deadhand
        self.auto_tpi = auto_tpi
        self.min_heater_temp = min_heater_temp
        self.max_heater_temp = max_heater_temp
        self.heater_temp_pb = max_heater_temp - min_heater_temp
        if self.auto_tpi:
            self.tpi_kp = 0
            self.tpi_kp = 0
        else:
            self.tpi_kp = tpi_kp
            self.tpi_ti = tpi_ti

        self.set_onoff_time1 = None
        self.set_onoff_time2 = None
        self.set_water_temp_time = None
        self.out_new = 0
        self.error_new = 0
        self.sensor_entity_id = sensor_entity_id
        self.ac_mode = ac_mode
        self.min_cycle_duration = min_cycle_duration
        self._cold_tolerance = cold_tolerance
        self._hot_tolerance = hot_tolerance
        self._keep_alive = keep_alive
        self._hvac_mode = initial_hvac_mode
        self._saved_target_temp = target_temp or next(iter(presets.values()), None)
        self._temp_precision = precision
        self._temp_target_temperature_step = target_temperature_step
        self._attr_hvac_modes = [HVACMode.HEAT, HVACMode.OFF]
        self._active = False
        self._cur_temp: float | None = None
        self._temp_lock = asyncio.Lock()
        self._min_temp = min_temp
        self._max_temp = max_temp
        self.cur_water_temp = 40
        self.heater_temp_input_entity_id = heater_temp_input_entity_id
        self._attr_preset_mode = PRESET_NONE
        self._target_temp = target_temp
        self._attr_temperature_unit = unit
        self._attr_unique_id = unique_id
        self._attr_supported_features = (
            ClimateEntityFeature.TARGET_TEMPERATURE
            | ClimateEntityFeature.TURN_OFF
            | ClimateEntityFeature.TURN_ON
        )
        if len(presets):
            self._attr_supported_features |= ClimateEntityFeature.PRESET_MODE
            self._attr_preset_modes = [PRESET_NONE] + list(presets.keys())
        else:
            self._attr_preset_modes = [PRESET_NONE]
        self._presets = presets
        _LOGGER.debug("__init__ called in tpi_thermstat")

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added."""
        try:
            mqtt_data = self.hass.data["mqtt"]
        except KeyError:
            _LOGGER.error("The MQTT integration is not available. Please ensure 'mqtt:' is configured in your Home Assistant configuration.")
            return
        await ClimateEntity.async_added_to_hass(self)
        _LOGGER.debug("async_added_to_hass called in tpi_thermstat")
        # Add listener
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.sensor_entity_id], self._async_sensor_changed
            )
        )
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.heater_switch_entity_id], self._async_switch_changed
            )
        )

        if self._keep_alive:
            _LOGGER.debug("keep alive enabled")
            async_track_time_interval(
                self.hass, self._async_control_heating, self._keep_alive
            )


        @callback
        def _async_startup(_: Event | None = None) -> None:
            """Init on startup."""
            sensor_state = self.hass.states.get(self.sensor_entity_id)
            if sensor_state and sensor_state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            ):
                self._async_update_temp(sensor_state)
                self.async_write_ha_state()
            switch_state = self.hass.states.get(self.heater_switch_entity_id)
            if switch_state and switch_state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            ):
                self.hass.create_task(self._check_switch_initial_state())

        if self.hass.state is CoreState.running:
            _async_startup()
        else:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _async_startup)

        # Check If we have an old state
        if (old_state := await self.async_get_last_state()) is not None:
            # If we have no initial temperature, restore
            if self._target_temp is None:
                # If we have a previously saved temperature
                if old_state.attributes.get(ATTR_TEMPERATURE) is None:
                    if self.ac_mode:
                        self._target_temp = self.max_temp
                    else:
                        self._target_temp = self.min_temp
                    _LOGGER.warning(
                        "Undefined target temperature, falling back to %s",
                        self._target_temp,
                    )
                else:
                    self._target_temp = float(old_state.attributes[ATTR_TEMPERATURE])
            if (
                self.preset_modes
                and old_state.attributes.get(ATTR_PRESET_MODE) in self.preset_modes
            ):
                self._attr_preset_mode = old_state.attributes.get(ATTR_PRESET_MODE)
            if not self._hvac_mode and old_state.state:
                self._hvac_mode = HVACMode(old_state.state)

        else:
            # No previous state, try and restore defaults
            if self._target_temp is None:
                if self.ac_mode:
                    self._target_temp = self.max_temp
                else:
                    self._target_temp = self.min_temp
            _LOGGER.warning(
                "No previously saved temperature, setting to %s", self._target_temp
            )

        # Set default state to off
        if not self._hvac_mode:
            self._hvac_mode = HVACMode.OFF

    def _prepare_subscribe_topics(self) -> None:  # noqa: C901
        """(Re)Subscribe to topics."""
        topics: dict[str, dict[str, Any]] = {}
    async def _subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        return None
    @staticmethod
    def config_schema() -> vol.Schema:
        """Return the config schema."""
        return DISCOVERY_SCHEMA
    @property
    def precision(self) -> float:
        """Return the precision of the system."""
        if self._temp_precision is not None:
            return self._temp_precision
        return super().precision

    @property
    def target_temperature_step(self) -> float:
        """Return the supported step of target temperature."""
        if self._temp_target_temperature_step is not None:
            return self._temp_target_temperature_step
        # if a target_temperature_step is not defined, fallback to equal the precision
        return self.precision
    
    @property
    def target_temperature(self) -> float | None:
        """Return the temperature we try to reach."""
        return self._target_temp
    
    @property
    def min_temp(self) -> float:
        """Return the minimum temperature."""
        if self._min_temp is not None:
            return self._min_temp

        # get default temp from super class
        return super().min_temp

    @property
    def max_temp(self) -> float:
        """Return the maximum temperature."""
        if self._max_temp is not None:
            return self._max_temp

        # Get default temp from super class
        return super().max_temp
        
    @property
    def current_temperature(self) -> float | None:
        """Return the sensor temperature."""
        return self._cur_temp

    @property
    def hvac_mode(self) -> HVACMode | None:
        """Return current operation."""
        return self._hvac_mode

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current running hvac operation if supported.

        Need to be one of CURRENT_HVAC_*.
        """
        if self._hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        if not self._is_device_active:
            return HVACAction.IDLE
        return HVACAction.HEATING

    @property
    def _is_device_active(self) -> bool | None:
        """If the toggleable device is currently active."""
        if not self.hass.states.get(self.heater_switch_entity_id):
            return None

        return self.hass.states.is_state(self.heater_switch_entity_id, STATE_ON)

    @property
    def target_temperature(self) -> float | None:
        """Return the temperature we try to reach."""
        return self._target_temp

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        if hvac_mode == HVACMode.HEAT:
            self._hvac_mode = HVACMode.HEAT
            await self._async_control_heating()

        elif hvac_mode == HVACMode.OFF:
            self._hvac_mode = HVACMode.OFF
            self.current_state = 0
            self.tpi_error_old = 0
            self.tpi_out_old = 0
            self.tpi_start = 1
            self.tpi_active = False
            self.last_deadhand_reason = None
            if self._is_device_active:
                await self._async_heater_turn_off(force=True)
        else:
            _LOGGER.error("Unrecognized hvac mode: %s", hvac_mode)
            return
        # Ensure we update the current operation after changing the mode
        self.async_write_ha_state()

    async def _async_sensor_changed(
        self, event: EventType[EventStateChangedData]
    ) -> None:
        """Handle temperature changes."""
        new_state = event.data["new_state"]
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        self._async_update_temp(new_state)
        await self._async_control_heating()
        self.async_write_ha_state()

    async def _check_switch_initial_state(self) -> None:
        """Prevent the device from keep running if HVACMode.OFF."""
        if self._hvac_mode == HVACMode.OFF and self._is_device_active:
            _LOGGER.warning(
                (
                    "The climate mode is OFF, but the switch device is ON. Turning off"
                    " device %s"
                ),
                self.heater_switch_entity_id,
            )
            await self._async_heater_turn_off()

    @callback
    def _async_switch_changed(self, event: EventType[EventStateChangedData]) -> None:
        """Handle heater switch state changes."""
        new_state = event.data["new_state"]
        old_state = event.data["old_state"]
        if new_state is None:
            return
        if old_state is None:
            self.hass.create_task(self._check_switch_initial_state())
        self.async_write_ha_state()

    @callback
    def _async_update_temp(self, state: State) -> None:
        """Update thermostat with latest state from sensor."""
        try:
            cur_temp = float(state.state)
            if not math.isfinite(cur_temp):
                raise ValueError(f"Sensor has illegal state {state.state}")
            self._cur_temp = cur_temp
        except ValueError as ex:
            _LOGGER.error("Unable to update from sensor: %s", ex)

    @property
    def precision(self) -> float:
        """Return the precision of the system."""
        if self._temp_precision is not None:
            return self._temp_precision
        return super().precision

    @property
    def target_temperature_step(self) -> float:
        """Return the supported step of target temperature."""
        if self._temp_target_temperature_step is not None:
            return self._temp_target_temperature_step
        # if a target_temperature_step is not defined, fallback to equal the precision
        return self.precision

    @property
    def current_temperature(self) -> float | None:
        """Return the sensor temperature."""
        return self._cur_temp

    @property
    def tpi_output(self) -> float | None:
        """Return the tpi output."""
        return self.out_new
    
    @property
    def tpi_state(self) -> float | None:
        """Return the tpi output."""
        return self.current_state
    
    @property
    def extra_state_attributes(self):
        """Return entity specific state attributes."""
        _attributes = {
            "tpi_state": self.current_state,
            "tpi_active": self.tpi_active,
            "deadhand": self.last_deadhand_reason,
            "heater_activate": self._is_device_active,
            "tpi_out_new": self.out_new,
            "tpi_out_old": self.tpi_out_old,
            "current_set_water_tempurature": self.cur_water_temp,
            "tpi_error_new": self.error_new, 
            "tpi_error_old": self.tpi_error_old,
            "is_tpi_start": self.tpi_start,
            "tpi_kp": self.tpi_kp, 
            "tpi_ti": self.tpi_ti
        }
        return _attributes
    
    @property
    def hvac_mode(self) -> HVACMode | None:
        """Return current operation."""
        return self._hvac_mode

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current running hvac operation if supported.

        Need to be one of CURRENT_HVAC_*.
        """
        if self._hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        if not self._is_device_active:
            return HVACAction.IDLE
        if self.ac_mode:
            return HVACAction.COOLING
        return HVACAction.HEATING
    
    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is None:
            return
        self._target_temp = temperature
        if self.auto_tpi:
            self.current_state = 0
        await self._async_control_heating()
        self.async_write_ha_state()


    async def _async_heater_turn_on(self, force=False) -> None:
        _LOGGER.debug("_async_heater_turn_on called")
        if self.set_onoff_time1 is not None and time.time() - self.set_onoff_time1 < 600 and not force:
            _LOGGER.info('althrough turn on/off heater, but to short time since last turn on/off, dismiss it')
            return
        """Turn heater toggleable device on."""
        data = {ATTR_ENTITY_ID: self.heater_switch_entity_id}
        await self.hass.services.async_call(
            HA_DOMAIN, SERVICE_TURN_ON, data, context=self._context
        )
        
        self.set_onoff_time1 = time.time()

    async def _async_heater_turn_off(self, force=False) -> None:
        """Turn heater toggleable device off."""
        _LOGGER.debug("_async_heater_turn_off called")
        if self.set_onoff_time2 is not None and time.time() - self.set_onoff_time2 < 600 and not force:
            _LOGGER.info('althrough turn on/off heater, but to short time since last turn on/off, dismiss it')
            return
        data = {ATTR_ENTITY_ID: self.heater_switch_entity_id}
        await self.hass.services.async_call(
            HA_DOMAIN, SERVICE_TURN_OFF, data, context=self._context
        )

        self.set_onoff_time2 = time.time()

    async def _async_heater_set_water_temperature(self, temp, force=False) -> None:
        if self.set_water_temp_time is not None and time.time() - self.set_water_temp_time < 180 and not force:
            _LOGGER.info('althrough set water temperature, but to short time since last operation, dismiss it')
            return
        
        temp = int(temp)
        
        if temp < self.min_heater_temp:
            temp = self.min_heater_temp
        if temp > self.max_heater_temp:
            temp = self.max_heater_temp
        self.set_water_temp_time = time.time()
        _LOGGER.debug('set water temperature %s %s',self.heater_temp_topic, str(temp) )
        if self.heater_temp_topic != 'not_set':
            payload = self.heater_temp_payload + str(temp)
            await self.async_publish(
                    self.heater_temp_topic,
                    payload
                )
        else:
            data = {ATTR_ENTITY_ID: self.heater_temp_input_entity_id, ATTR_VALUE: temp}
            await self.hass.services.async_call(
                INPUT_TEXT_DOMAIN, SERVICE_SET_VALUE, data, context=self._context
            )

        

    async def _async_control_heating(
        self, pt: datetime | None = None, force: bool = False
    ) -> None:

        """Check if we need to turn heating on or off."""
        _LOGGER.debug("entering _async_control_heating")
        async with self._temp_lock:
            _LOGGER.debug("running heating control once, TPI state %s, tpi_active %s , deadhand %s, heater activate %s", self.current_state, self.tpi_active, self.last_deadhand_reason, self._is_device_active)
            _LOGGER.debug("_async_control_heating lock accquired")
            if not self._active and None not in (
                self._cur_temp,
                self._target_temp,
            ):
                self._active = True
                _LOGGER.info(
                    (
                        "Obtained current and target temperature. "
                        "tpi thermostat active. %s, %s"
                    ),
                    self._cur_temp,
                    self._target_temp,
                )

            if (not self._active) or self._hvac_mode == HVACMode.OFF:
                _LOGGER.debug("heating control not active")
                return


            assert self._cur_temp is not None and self._target_temp is not None
            # deadhand asserts
            too_cold = (self._target_temp >= self._cur_temp + self.deadhand / 2) and self.last_deadhand_reason is None
            too_hot = (self._cur_temp >= self._target_temp + self.deadhand / 2) and self.last_deadhand_reason is None

            if too_hot:
                _LOGGER.info("Reach High DeadHand, Turning off heater %s", self.heater_switch_entity_id)
                self.tpi_active = False
                self.last_deadhand_reason = 'too_hot'
                if self._is_device_active:
                    await self._async_heater_turn_off(force=True)

            elif too_cold:
                _LOGGER.info("Reach Low DeadHand, Turning on heater %s", self.heater_switch_entity_id)
                self.tpi_active = False
                self.last_deadhand_reason = 'too_cold'
                await self._async_heater_turn_on(force=True)
                await self._async_heater_set_water_temperature(self.max_heater_temp - 5, force=True)
            
            elif self.last_deadhand_reason is None:
                self.tpi_active = True
            

            # restore from deadhand
            restore_from_hot = self._target_temp >= self._cur_temp 
            restore_from_cold = self._cur_temp >= self._target_temp 
            if self.tpi_active == False and self.last_deadhand_reason == 'too_hot' and restore_from_hot:
                self.tpi_active = True
                self.last_deadhand_reason = None
                self.current_state = 0
                _LOGGER.info("Recover from deadhand too_hot.")
                _LOGGER.info("force state 0")
            if self.tpi_active == False and self.last_deadhand_reason == 'too_cold' and restore_from_cold:
                self.tpi_active = True
                self.last_deadhand_reason = None
                self.current_state = 0
                _LOGGER.info("Recover from deadhand too_cold.")
                _LOGGER.info("force state 0")
            # start tpi state cycle
            if self.tpi_active:
                # need calc kp and ti
                if self.current_state == 0:
                    if self.auto_tpi == False:
                        _LOGGER.info("No need auto calc tpi, entering state 4 from state 0.")

                        self.tpi_start = 1
                        self.current_state = 4
                        
                        return
                    else:
                        self.current_state_start_calc_time = None
                        self.current_state_start_calc_temp = None
                        self.current_state_end_calc_time = None
                        self.current_state_end_calc_temp = None
                        await self._async_heater_set_water_temperature(self.max_heater_temp - 5, force=True)
                        if self._cur_temp <= self._target_temp - self._cold_tolerance:
                            _LOGGER.info("entering state 1 from state 0.")

                            self.current_state = 1
                            
                            return
                        if self._cur_temp >= self._target_temp + self._hot_tolerance:
                            _LOGGER.info("entering state 2 from state 0.")
                            self.current_state = 2
                            
                            return
                        if self._cur_temp < self._target_temp + self._hot_tolerance and  self._cur_temp > self._target_temp - self._cold_tolerance:
                            _LOGGER.info("entering state 3 from state 0.")
                            self.current_state = 3
                            
                            return
                ## Turn ON until it reaches upper limit of proportional band (Set point +PB/2), Then Turn OFF until is it reaches lower limit of proportional band
                if self.current_state == 1:
                    if not self._is_device_active:
                        await self._async_heater_turn_on()
                    if self._cur_temp >= self._target_temp - self._cold_tolerance and self.current_state_start_calc_time is None:
                        self.current_state_start_calc_time = time.time()
                        self.current_state_start_calc_temp = self._cur_temp
                        _LOGGER.info("state 1 recorded start_calc_time %s, start_calc_temp %s.", self.current_state_start_calc_time, self.current_state_start_calc_temp )
                        
                        return
                    if self._cur_temp >= self._target_temp + self._hot_tolerance and self.current_state_start_calc_time is not None and self.current_state_end_calc_time is None:
                        self.current_state_end_calc_time = time.time()
                        self.current_state_end_calc_temp = self._cur_temp
                        _LOGGER.info("state 1 recorded end_calc_time %s, end_calc_temp %s.", self.current_state_end_calc_time, self.current_state_end_calc_temp )

                        _LOGGER.info("entering state 5 from state 1")
                        self.current_state = 5
                        
                        return

                ## Turn OFF until is it reaches lower limit of proportional band,Turn ON until it reaches upper limit of proportional band ( Set point +PB/2
                if self.current_state == 2 or self.current_state == 3:
                    if self._is_device_active and self.current_state_start_calc_time is None:
                        await self._async_heater_turn_off()
                        
                        return
                    if self._cur_temp <= self._target_temp - self._cold_tolerance and self.current_state_start_calc_time is None:
                        self.current_state_start_calc_time = time.time()
                        self.current_state_start_calc_temp = self._cur_temp
                        _LOGGER.info("state 2 recorded start_calc_time %s, start_calc_temp %s.", self.current_state_start_calc_time, self.current_state_start_calc_temp )

                        if not self._is_device_active:
                            await self._async_heater_turn_on()
                        
                        return
                    if self._cur_temp >= self._target_temp + self._hot_tolerance and self.current_state_start_calc_time is not None and self.current_state_end_calc_time is None:
                        self.current_state_end_calc_time = time.time()
                        self.current_state_end_calc_temp = self._cur_temp
                        _LOGGER.info("state 2 recorded end_calc_time %s, end_calc_temp %s.", self.current_state_end_calc_time, self.current_state_end_calc_temp )

                        self.current_state = 5
                        _LOGGER.info("entering state 5 from state 2 or 3")
                        
                        return
                ## calc tpi cycle and wait cycle
                if self.current_state == 5:
                    if self.auto_tpi and self.current_state_start_calc_time > 0 and self.current_state_start_calc_temp > 0 and self.current_state_end_calc_temp > 0 and self.current_state_end_calc_time > 0:
                        ## need calc tpi
                        slope = int(10000 * math.fabs((self.current_state_end_calc_temp-self.current_state_start_calc_temp)/(self.current_state_end_calc_time-self.current_state_start_calc_time)))
                        if slope > SLOPE_TABLE_MAX_SLOPE:
                            slope = SLOPE_TABLE_MAX_SLOPE
                        if slope < SLOPE_TABLE_START_SLOPE:
                            slope = SLOPE_TABLE_START_SLOPE
                        
                        self.tpi_kp = SLOPE_TABLE_START_KP - (slope - SLOPE_TABLE_START_SLOPE) * SLOPE_TABLE_STEP_KP
                        self.tpi_ti = SLOPE_TABLE_START_TI - (slope - SLOPE_TABLE_START_SLOPE) * SLOPE_TABLE_STEP_TI
                        self.current_state_start_calc_time = None
                        self.current_state_start_calc_temp = None
                        self.current_state_end_calc_time = None
                        self.current_state_end_calc_temp = None
                        _LOGGER.info("recalc kp %s and ti %s, with slope %s", self.tpi_kp, self.tpi_ti, slope)

                    if self._is_device_active:
                        await self._async_heater_turn_off()
                        
                        return

                    if self._cur_temp <= self._target_temp:
                        if not self._is_device_active:
                            await self._async_heater_turn_on()
                        self.tpi_start = 1
                        self.current_state = 4
                        
                        return
                ## real tpi cycle
                if self.current_state == 4:
                    if not self._is_device_active:
                        await self._async_heater_turn_on()
                    self.error_new = (self._target_temp - self._cur_temp) / (self._hot_tolerance+self._cold_tolerance)
                    if self.tpi_start == 1:
                        self.out_new = 0.5 + self.tpi_kp * self.error_new
                    else:
                        if self.tpi_out_old > 2.0:
                            self.tpi_out_old = 2.0
                        if self.tpi_out_old < -1:
                            self.tpi_out_old = -1
                        self.out_new = self.tpi_out_old + self.tpi_kp*(self.error_new - self.tpi_error_old) + (self.tpi_kp * 60 * self.error_new)/self.tpi_ti
                    _LOGGER.info("tpi cycle output: out_new %s, out_old %s, error_new %s, error_old %s, tpi_start %s, tpi_kp: %s, tpi_ti: %s", self.out_new, self.tpi_out_old, self.error_new, self.tpi_error_old, self.tpi_start, self.tpi_kp, self.tpi_ti)
                    self.tpi_start = 0
                    self.tpi_error_old =self.error_new
                    self.tpi_out_old = self.out_new


                    if self.out_new > 1:
                        self.out_new = 1
                    elif self.out_new < 0:
                        self.out_new = 0    
                    
                    ## trans tpi out to water temp
                    self.cur_water_temp = self.min_heater_temp + self.heater_temp_pb*self.out_new
                    _LOGGER.info("setting temp from tpi, %s" , self.cur_water_temp)
                    
                    await self._async_heater_set_water_temperature(self.cur_water_temp)
