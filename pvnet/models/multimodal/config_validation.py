"""Validation functions for Multimodal configuration using modern type hints."""

import logging
from typing import Any, Type

logger = logging.getLogger(__name__)


def _check_key(
    cfg: dict[str, Any],
    key: str,
    required: bool = True,
    expected_type: Type | None = None,
    warn_on_type_mismatch: bool = False,
    context: str = "Configuration",
) -> None:
    """
    Checks for the presence and optionally the type of a key in the config dictionary.

    Logs a warning on type mismatch if warn_on_type_mismatch is True, otherwise
    raises a TypeError.

    Args:
        cfg: The configuration dictionary.
        key: The key to check within the dictionary.
        required: If True, raises KeyError if the key is missing. Defaults to True.
        expected_type (Type | None): The expected type object (e.g., str, int, dict)
            for the key's value. If None, type is not checked. Defaults to None.
        warn_on_type_mismatch: If True and expected_type is provided, logs a
            warning on type mismatch instead of raising TypeError. Defaults to False.
        context: String describing the context for error/warning messages
            (e.g., the name of the configuration section being checked).
            Defaults to "Configuration".

    Raises:
        KeyError: If `required` is True and the `key` is missing from `cfg`.
        TypeError: If `expected_type` is provided, the value associated with `key`
                   in `cfg` does not match the `expected_type`, and
                   `warn_on_type_mismatch` is False.
    """
    if key not in cfg:
        if required:
            raise KeyError(f"{context} missing required key: '{key}'")
        else:
            return

    value: Any = cfg[key]
    if expected_type is not None and not isinstance(value, expected_type):
        message = (
            f"{context} key '{key}' expected type {expected_type.__name__}, "
            f"found {type(value).__name__}."
        )
        if warn_on_type_mismatch:
            logger.warning(message)
        else:
            raise TypeError(message)


def _check_dict_section(
    cfg: dict[str, Any],
    section_name: str,
    required: bool = True,
    check_target: bool = True,
    check_sub_items_target: bool = False,
) -> dict[str, Any] | None:
    """
    Validates a configuration section expected to be a dictionary.

    Optionally checks for the presence of a '_target_' key within the section
    or within sub-dictionaries of the section. Logs a warning if an optional
    section is present but empty.

    Args:
        cfg: The main configuration dictionary.
        section_name: The key of the section to validate within `cfg`.
        required: If True, raises KeyError if the section is missing or empty
                  when `check_target` is also True. Defaults to True.
        check_target: If True, checks for a '_target_' key directly within the
                      section dictionary (if `check_sub_items_target` is False).
                      Defaults to True.
        check_sub_items_target: If True, assumes the section dictionary contains
                                further dictionaries as values and checks for
                                '_target_' within those sub-dictionaries.
                                Defaults to False.

    Returns:
        dict[str, Any] | None: The section dictionary if validation passes.
            Returns None if the section is optional (`required=False`) and is
            missing from `cfg`.

    Raises:
        KeyError: If `required` is True and the section `section_name` is missing
                  from `cfg`, or if `check_target` or `check_sub_items_target`
                  is True and the required '_target_' key(s) are missing.
        TypeError: If the section identified by `section_name` is not a dictionary,
                   or if `check_sub_items_target` is True and any sub-item is
                   not a dictionary.
    """
    if section_name not in cfg:
        if required:
            raise KeyError(f"Configuration missing required section: '{section_name}'")
        else:
            return None

    section_content: Any = cfg.get(section_name)

    if not isinstance(section_content, dict):
        raise TypeError(
            f"Config section '{section_name}' must be a dictionary, "
            f"found {type(section_content).__name__}."
        )

    if not section_content:
        if not required:
            logger.warning(f"Optional config section '{section_name}' is present but empty.")
        elif required and (check_target or check_sub_items_target):
             raise KeyError(
                 f"Required config section '{section_name}' is empty and requires sub-key(s) "
                 f"like '_target_'."
             )
        return section_content

    if check_sub_items_target:
        for source, sub_config in section_content.items():
            if not isinstance(sub_config, dict):
                raise TypeError(
                    f"Config for source '{source}' in '{section_name}' "
                    f"must be a dictionary, found {type(sub_config).__name__}."
                )
            if "_target_" not in sub_config:
                raise KeyError(
                    f"Source '{source}' in section '{section_name}' "
                    f"missing required sub-key: '_target_'"
                )
    elif check_target:
        if "_target_" not in section_content:
            raise KeyError(
                f"Config section '{section_name}' is missing required "
                f"sub-key: '_target_'"
            )

    return section_content


def _check_time_parameter(
    cfg: dict[str, Any],
    param_name: str,
    owner_key: str,
    required_if_owner_present: bool = True,
) -> None:
    """
    Checks a time parameter potentially associated with an optional feature/encoder.

    Verifies the presence and type (expects int, warns otherwise) of a time
    parameter (`param_name`) if the feature enabling it (`owner_key`) is present
    and truthy in the configuration.

    Args:
        cfg: The main configuration dictionary.
        param_name: The name of the time parameter key (e.g., 'sat_history_minutes').
        owner_key: The key indicating the associated feature is enabled
                   (e.g., 'sat_encoder'). The check proceeds if `cfg.get(owner_key)`
                   evaluates to True.
        required_if_owner_present: If True, raises KeyError if the `owner_key` is
                                   present/truthy but the `param_name` key is
                                   missing from `cfg`. If False, logs a warning
                                   instead. Defaults to True.

    Raises:
        KeyError: If the `owner_key` is present/truthy in `cfg`,
                  `required_if_owner_present` is True, and the `param_name` key
                  is missing from `cfg`.
    """
    if owner_key in cfg and cfg.get(owner_key):
        context = f"Config includes '{owner_key}'"
        if param_name not in cfg:
            message = f"{context} but is missing '{param_name}'."
            if required_if_owner_present:
                raise KeyError(message)
            else:
                logger.warning(f"{message} (Action: May use default value if applicable).")
        else:
            _check_key(
                cfg,
                param_name,
                required=False,
                expected_type=int,
                warn_on_type_mismatch=True,
                context=f"Parameter '{param_name}' associated with '{owner_key}'",
            )


def _validate_nwp_specifics(cfg: dict[str, Any], nwp_section: dict[str, Any]) -> None:
    """
    Performs validation specific to the NWP (Numerical Weather Prediction) configuration.

    Checks for the presence and structure of required NWP time parameters
    ('nwp_history_minutes', 'nwp_forecast_minutes') and ensures their keys
    match the sources defined in the 'nwp_encoders_dict'. Logs warnings for
    an empty NWP source list or non-integer time values.

    Args:
        cfg: The main configuration dictionary.
        nwp_section: The validated 'nwp_encoders_dict' section content (must be
                     a dictionary, potentially empty).

    Raises:
        KeyError: If required NWP time parameter keys ('nwp_history_minutes',
                  'nwp_forecast_minutes') are missing when `nwp_section` has sources.
        TypeError: If 'nwp_history_minutes' or 'nwp_forecast_minutes' exist but
                   are not dictionaries.
        ValueError: If the keys within 'nwp_history_minutes' or
                    'nwp_forecast_minutes' do not exactly match the keys (NWP sources)
                    present in `nwp_section`.
    """
    nwp_sources: list[str] = list(nwp_section.keys())
    if not nwp_sources:
        logger.warning("'nwp_encoders_dict' is defined but contains no NWP sources.")
        return

    context = f"Config includes 'nwp_encoders_dict' with sources: {nwp_sources}"
    _check_key(cfg, "nwp_history_minutes", required=True, expected_type=dict, context=context)
    _check_key(cfg, "nwp_forecast_minutes", required=True, expected_type=dict, context=context)

    nwp_hist_times: dict[str, Any] = cfg["nwp_history_minutes"]
    nwp_forecast_times: dict[str, Any] = cfg["nwp_forecast_minutes"]

    hist_keys: set[str] = set(nwp_hist_times.keys())
    forecast_keys: set[str] = set(nwp_forecast_times.keys())
    encoder_keys: set[str] = set(nwp_sources)

    if hist_keys != encoder_keys:
        missing_in_hist = encoder_keys - hist_keys
        extra_in_hist = hist_keys - encoder_keys
        raise ValueError(
            f"Keys in 'nwp_history_minutes' {hist_keys} do not match sources "
            f"in 'nwp_encoders_dict' {encoder_keys}. "
            f"Missing: {missing_in_hist}, Extra: {extra_in_hist}"
        )
    if forecast_keys != encoder_keys:
        missing_in_forecast = encoder_keys - forecast_keys
        extra_in_forecast = forecast_keys - encoder_keys
        raise ValueError(
            f"Keys in 'nwp_forecast_minutes' {forecast_keys} do not match sources "
            f"in 'nwp_encoders_dict' {encoder_keys}. "
            f"Missing: {missing_in_forecast}, Extra: {extra_in_forecast}"
        )

    for source, value in nwp_hist_times.items():
        if not isinstance(value, int):
            logger.warning(
                f"'nwp_history_minutes' for source '{source}' expected int, "
                f"found {type(value).__name__}."
            )
    for source, value in nwp_forecast_times.items():
        if not isinstance(value, int):
            logger.warning(
                f"'nwp_forecast_minutes' for source '{source}' expected int, "
                f"found {type(value).__name__}."
            )


def validate_multimodal_config(cfg: dict[str, Any]) -> dict[str, bool]:
    """
    Validates the configuration dictionary for the Multimodal Model.

    Checks that the configuration dictionary `cfg` has the expected structure,
    keys, and basic types required for the Multimodal Model. It utilizes
    helper functions for granular checks. Critical issues (missing required keys
    like '_target_', incorrect types for major sections, mismatched NWP keys)
    will raise an appropriate exception (`KeyError`, `TypeError`, `ValueError`).
    Non-critical issues (e.g., optional sections present but empty, expected
    integer types being incorrect but tolerated) will be logged as warnings.

    Args:
        cfg (dict[str, Any]): The configuration dictionary to validate. Expected
            to be a mapping from string keys to various value types.

    Returns:
        dict[str, bool]: A dictionary indicating success: `{"valid": True}`.
            If validation fails due to a critical issue, an exception is raised
            instead of returning normally.

    Raises:
        TypeError: If a configuration section expected to be a dictionary is not,
                   or if required time parameters (e.g., for NWP) are not dictionaries
                   when expected, or if a value has an incorrect type and
                   `warn_on_type_mismatch` was False for that check.
        KeyError: For critical validation failures like missing required top-level
                  keys (e.g., '_target_', 'forecast_minutes', 'history_minutes',
                  'output_network', 'optimizer'), missing required '_target_'
                  sub-keys within sections, or missing required time parameters
                  when their associated feature is enabled.
        ValueError: If keys in NWP time parameter dictionaries ('nwp_history_minutes',
                    'nwp_forecast_minutes') do not exactly match the source keys
                    defined in 'nwp_encoders_dict'.
    """
    _check_key(cfg, "_target_", required=True, expected_type=str)
    _check_key(
        cfg, "forecast_minutes", required=True, expected_type=int, warn_on_type_mismatch=True
    )
    _check_key(
        cfg, "history_minutes", required=True, expected_type=int, warn_on_type_mismatch=True
    )

    _check_dict_section(cfg, "output_network", required=True, check_target=True)
    _check_dict_section(cfg, "optimizer", required=True, check_target=True)

    _check_dict_section(cfg, "sat_encoder", required=False, check_target=True)
    _check_dict_section(cfg, "pv_encoder", required=False, check_target=True)
    _check_dict_section(cfg, "sensor_encoder", required=False, check_target=True)

    _check_time_parameter(
        cfg, "sat_history_minutes", owner_key="sat_encoder", required_if_owner_present=True
    )
    _check_time_parameter(
        cfg, "pv_history_minutes", owner_key="pv_encoder", required_if_owner_present=True
    )
    _check_time_parameter(
        cfg, "sensor_history_minutes", owner_key="sensor_encoder", required_if_owner_present=False
    )
    _check_time_parameter(
        cfg, "sensor_forecast_minutes", owner_key="sensor_encoder", required_if_owner_present=False
    )

    nwp_section = _check_dict_section(
        cfg, "nwp_encoders_dict", required=False, check_sub_items_target=True
    )

    if nwp_section is not None:
        _validate_nwp_specifics(cfg, nwp_section)

    return {"valid": True}
