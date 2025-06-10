"""Validation functions for Multimodal configuration."""

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
    """Check if a key exists in a dictionary and optionally validate its type.

    Args:
        cfg: The configuration dictionary.
        key: The key to check.
        required: If True, raise error if key is missing.
        expected_type: The expected type or tuple of types for the key's value.
        warn_on_type_mismatch: If True, log warning on type mismatch instead of raising.
        context: Context for error/warning messages.
    """
    if key not in cfg:
        if required:
            raise KeyError(f"{context} missing required key: '{key}'")
        return

    value: Any = cfg[key]
    
    if expected_type is not None and not isinstance(value, expected_type):
        message = (
            f"{context} key '{key}' expected type {expected_type}, "
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
) -> dict[str, Any] | None:
    """Validate a config section, ensuring it's a dict and optionally checking targets.

    Args:
        cfg: The main configuration dictionary.
        section_name: Key of the section to validate.
        required: If True, raise error if section is missing.
        check_target: If True, check for '_target_' key within the section dict.
        check_sub_items_target: If True, check for '_target_' within sub-dictionaries.

    Returns:
        The section dictionary if valid, or None if optional and missing.

    Raises:
        KeyError: If required section/target is missing.
        TypeError: If section or sub-items are not dictionaries as expected.
    """
    if section_name not in cfg:
        if required:
            raise KeyError(f"Configuration missing required section: '{section_name}'")
        else:
            return None

    section_content: Any = cfg.get(section_name)

    # Ensure section follows dict structure
    if not isinstance(section_content, dict):
        raise TypeError(
            f"Config section '{section_name}' must be a dictionary, "
            f"found {type(section_content).__name__}."
        )

    if not section_content and not required:
        logger.warning(
            f"Optional config section '{section_name}' is present but empty."
        )
        return section_content

    if check_target:
         if section_content or required:
              if "_target_" not in section_content:
                  raise KeyError(
                      f"Config section '{section_name}' is missing required "
                      f"sub-key: '_target_'"
                  )

    return section_content


def _validate_nwp_specifics(cfg: dict[str, Any], nwp_section: dict[str, Any]) -> None:
    """Validate NWP-specific time/interval dictionaries against NWP sources.

    Args:
        cfg: The main configuration dictionary.
        nwp_section: The validated 'nwp_encoders_dict' section content.

    Raises:
        KeyError: If required NWP time/interval parameter dictionaries are missing.
        TypeError: If NWP time/interval parameters are not dictionaries.
        ValueError: If keys in time/interval dicts do not match NWP sources.
    """
    nwp_sources: list[str] = list(nwp_section.keys())
    if not nwp_sources:
        logger.warning("'nwp_encoders_dict' is defined but contains no NWP sources.")
        return

    context = f"Config includes 'nwp_encoders_dict' with sources: {nwp_sources}"

    # Check time param values are integers
    _check_key(
        cfg, "nwp_history_minutes", required=True, expected_type=dict, context=context
    )
    _check_key(
        cfg, "nwp_forecast_minutes", required=True, expected_type=dict, context=context
    )
    _check_key(
        cfg, "nwp_interval_minutes", required=True, expected_type=dict, context=context
    )

    nwp_hist_times: dict[str, Any] = cfg["nwp_history_minutes"]
    nwp_forecast_times: dict[str, Any] = cfg["nwp_forecast_minutes"]

    hist_keys: set[str] = set(nwp_hist_times.keys())
    forecast_keys: set[str] = set(nwp_forecast_times.keys())
    encoder_keys: set[str] = set(nwp_sources)

    # Verify time params are provided for specifically defined NWP sources
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

    _check_key(
        cfg, "nwp_interval_minutes", required=True, expected_type=dict, context=context
    )
    nwp_intervals: dict[str, int] = cfg["nwp_interval_minutes"]
    interval_keys: set[str] = set(nwp_intervals.keys())
    if interval_keys != encoder_keys:
        missing_in_interval = encoder_keys - interval_keys
        extra_in_interval = interval_keys - encoder_keys
        raise ValueError(
            f"Keys in 'nwp_interval_minutes' {interval_keys} do not match sources "
            f"in 'nwp_encoders_dict' {encoder_keys}. "
            f"Missing: {missing_in_interval}, Extra: {extra_in_interval}"
        )

    # Check time param values are integers
    for dict_name in ["nwp_history_minutes", "nwp_forecast_minutes", "nwp_interval_minutes"]:
        param_dict = cfg[dict_name]
        for source_key in param_dict:
            _check_key(
                param_dict,
                source_key,
                expected_type=int,
                context=f"Dictionary '{dict_name}'",
            )


def _check_output_quantiles_config(
    cfg: dict[str, Any], context: str = "Top Level"
) -> None:
    """Validate the 'output_quantiles' parameter in the configuration.

    Args:
        cfg: The configuration dictionary.
        context: Context for error messages.

    Raises:
        KeyError: If 'output_quantiles' key is missing.
        TypeError: If 'output_quantiles' is not a list/tuple or contains non-numbers.
        ValueError: If 'output_quantiles' is empty.
    """
    _check_key(
        cfg,
        "output_quantiles",
        required=True,
        expected_type=(list, tuple),
        context=context,
    )
    quantiles = cfg["output_quantiles"]

    for i, q_value in enumerate(quantiles):
        if not isinstance(q_value, (int, float)):
            raise TypeError(
                f"{context}: Element {i} in 'output_quantiles' must be a number, "
                f"found {type(q_value).__name__}."
            )


def get_encoder_config(
    cfg: dict[str, Any], section_key: str, context: str, source_key: str | None = None
) -> dict[str, Any]:
    """Retrieve an encoder's configuration dictionary (handles nesting).

    Args:
        cfg: The main configuration dictionary.
        section_key: Key pointing to the encoder config or dict of configs.
        context: Context string for error messages.
        source_key: Optional sub-key for nested configs (like NWP sources).

    Returns:
        The specific encoder's configuration dictionary.

    Raises:
        KeyError: If `section_key` or `source_key` (if provided) is not found.
        TypeError: If the retrieved configuration is not a dictionary.
    """
    encoder_config: dict[str, Any] | None = None
    section_dict = cfg.get(section_key)

    if not isinstance(section_dict, dict):
        if section_dict is not None:
            raise TypeError(
                f"{context}: Section '{section_key}' is not a valid dictionary, "
                f"found {type(section_dict).__name__}."
            )
        elif source_key:
            raise KeyError(
                f"{context}: Cannot find section '{section_key}' to retrieve source '{source_key}'."
            )
        else:
            raise KeyError(f"{context}: Cannot find section '{section_key}'.")

    if source_key:
        encoder_config = section_dict.get(source_key)
        if not isinstance(encoder_config, dict):
            raise TypeError(
                f"{context}: Config for source '{source_key}' in '{section_key}' "
                f"must be a dictionary, found {type(encoder_config).__name__}."
            )
    else:
        encoder_config = section_dict

    if not isinstance(encoder_config, dict):
        # This case should technically be caught by the first isinstance check if source_key is None
        # but kept for robustness.
        raise TypeError(
            f"{context}: Final resolved encoder config is not a dictionary."
        )

    return encoder_config


def validate_static_config(cfg: dict[str, Any]) -> None:
    """Perform static validation of the multimodal configuration dictionary.

    Checks presence, types, and basic constraints of core config parameters
    and sections before checking batch data.

    Args:
        cfg: The multimodal configuration dictionary to validate.

    Raises:
        KeyError, TypeError, ValueError: If static configuration rules are violated.
    """
    _check_output_quantiles_config(cfg, context="Top Level")
    _check_key(
        cfg,
        "forecast_minutes",
        required=True,
        expected_type=int,
        warn_on_type_mismatch=True,
    )
    _check_key(
        cfg,
        "history_minutes",
        required=True,
        expected_type=int,
        warn_on_type_mismatch=True,
    )
    _check_key(
        cfg,
        "min_sat_delay_minutes",
        required=False,
        expected_type=int,
        warn_on_type_mismatch=True,
    )
    _check_key(cfg, "nwp_interval_minutes", required=False, expected_type=dict)
    _check_key(cfg, "embedding_dim", required=False, expected_type=int)
    _check_key(cfg, "include_sun", required=False, expected_type=bool)
    _check_key(cfg, "include_gsp_yield_history", required=False, expected_type=bool)
    _check_key(cfg, "add_image_embedding_channel", required=False, expected_type=bool)

    _check_dict_section(cfg, "output_network", required=True, check_target=True)
    _check_dict_section(cfg, "optimizer", required=True, check_target=True)

    # Satellite Encoder
    sat_section = _check_dict_section(
        cfg, "sat_encoder", required=False, check_target=True
    )
    if sat_section:
        _check_key(
            cfg,
            "sat_history_minutes",
            required=True,
            expected_type=int,
            context="Config with 'sat_encoder'",
        )
        if "required_parameters" in sat_section:
            for param in sat_section["required_parameters"]:
                _check_key(sat_section, param, required=True, expected_type=int, context="sat_encoder")
                if sat_section.get(param, 0) <= 0:
                    raise ValueError(f"sat_encoder: Parameter '{param}' must be positive integer.")

    # NWP Encoders
    nwp_section = _check_dict_section(
        cfg, "nwp_encoders_dict", required=False, check_target=False
    )
    if nwp_section is not None:
        _validate_nwp_specifics(cfg, nwp_section)
        for source, source_specific_encoder_config in nwp_section.items():
            if not isinstance(source_specific_encoder_config, dict):
                raise TypeError(
                    f"Config for NWP source '{source}' in 'nwp_encoders_dict' "
                    f"must be a dictionary, found "
                    f"{type(source_specific_encoder_config).__name__}."
                )

            _check_key(source_specific_encoder_config, "_target_", required=True,
                       context=f"Config for NWP source '{source}'")

            if "required_parameters" in source_specific_encoder_config:
                context = f"nwp_encoders_dict[{source}]"
                for param in source_specific_encoder_config["required_parameters"]:
                    _check_key(source_specific_encoder_config, param, required=True, 
                               expected_type=int, context=context)
                    if source_specific_encoder_config.get(param, 0) <= 0:
                        raise ValueError(f"{context}: Parameter '{param}' must be positive integer.")
