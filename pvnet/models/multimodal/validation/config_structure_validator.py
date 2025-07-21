"""Validation functions for Multimodal configuration."""

from typing import Any, Type


def _check_key(
    cfg: dict[str, Any],
    key: str,
    required: bool = True,
    expected_type: Type | None = None,
    context: str = "Configuration",
) -> None:
    """Check if a key exists in the configuration and optionally validate its type.

    Args:
        cfg: The configuration dictionary.
        key: The key to check.
        required: If True, raise error if key is missing.
        expected_type: The expected type or tuple of types for the key's value.
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
        raise TypeError(message)


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
        raise ValueError("'nwp_encoders_dict' is defined but contains no NWP sources.")
        return

    context = (
        f"Config includes 'nwp_encoders_dict' with sources: {nwp_sources}. "
        "NWP time parameters must be defined for all specified sources."
    )

    # Check that time and interval param sections are dicts
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

    # Check that values within time interval dicts are integers
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
    cfg: dict[str, Any], context: str = "Main Model Configuration"
) -> None:
    """Validate the 'output_quantiles' parameter in the configuration.

    Args:
        cfg: The configuration dictionary.
        context: Context for error messages.

    Raises:
        TypeError: If 'output_quantiles' is present but not a list / tuple / None.
        ValueError: If 'output_quantiles' is an empty list / tuple.
    """
    _check_key(
        cfg,
        "output_quantiles",
        required=False,
        expected_type=(list, tuple, type(None)),
        context=context,
    )
    quantiles = cfg.get("output_quantiles")

    if quantiles is None:
        return

    if not quantiles:
        raise ValueError(
            f"{context}: 'output_quantiles' cannot be empty list or tuple if provided."
        )

    for i, q_value in enumerate(quantiles):
        if not isinstance(q_value, (int, float)):
            raise TypeError(
                f"{context}: Element {i} in 'output_quantiles' must be number, "
                f"found {type(q_value).__name__}."
            )


def validate_model_config(cfg: dict[str, Any]) -> None:
    """Perform static validation of the multimodal configuration dictionary.

    Checks presence, types, and basic constraints of core config parameters
    and sections before checking batch data. Main five stages as follows:

    1. Checks Output Quantiles: Ensures 'output_quantiles' is present,
       is a list or tuple, and contains only numerical values.

    2. Checks Core Keys: Validates presence and type of parameters such as
       'forecast_minutes', 'history_minutes', and optional parameters like
       'min_sat_delay_minutes', 'embedding_dim', 'include_sun',
       'include_gsp_yield_history', and 'add_image_embedding_channel'.

    3. Checks Output Network and Optimizer: Verifies that 'output_network'
       and 'optimizer' sections exist and each contains a '_target_' key
       for component instantiation.

    4. Checks Satellite Encoder: If 'sat_encoder' is present, it validates
       its structure, including '_target_' key, 'sat_history_minutes',
       and positive integer 'required_parameters'.

    5. Checks NWP Encoders: If 'nwp_encoders_dict' is present, it performs validation:
       Ensures 'nwp_history_minutes', 'nwp_forecast_minutes', and
       'nwp_interval_minutes' are dictionaries whose keys match the NWP sources
       in 'nwp_encoders_dict'.

       Confirms values within these time/interval dictionaries are integers.
       For each NWP source encoder config, it validates '_target_' key and
       checks for positive integer 'required_parameters'.

    Args:
        cfg: The multimodal configuration dictionary to validate.

    Raises:
        KeyError, TypeError, ValueError: If static configuration rules are violated.
    """
    _check_output_quantiles_config(cfg, context="Main Model Configuration")
    _check_key(
        cfg,
        "forecast_minutes",
        required=True,
        expected_type=int,
    )
    _check_key(
        cfg,
        "history_minutes",
        required=True,
        expected_type=int,
    )
    _check_key(cfg, "embedding_dim", required=False, expected_type=int)
    _check_key(cfg, "include_sun", required=False, expected_type=bool)
    _check_key(cfg, "include_gsp_yield_history", required=False, expected_type=bool)
    _check_key(cfg, "add_image_embedding_channel", required=False, expected_type=bool)

    _check_key(cfg, "output_network", required=True, expected_type=dict)
    _check_key(cfg["output_network"], "_target_", required=True, context="Section 'output_network'")
    _check_key(cfg, "optimizer", required=True, expected_type=dict)
    _check_key(cfg["optimizer"], "_target_", required=True, context="Section 'optimizer'")

    # Satellite Encoder
    _check_key(cfg, "sat_encoder", required=False, expected_type=dict)
    if sat_section := cfg.get("sat_encoder"):
        _check_key(
            cfg,
            "min_sat_delay_minutes",
            required=False,
            expected_type=int,
        )
        _check_key(sat_section, "_target_", required=True, context="Section 'sat_encoder'")
        _check_key(
            cfg,
            "sat_history_minutes",
            required=True,
            expected_type=int,
            context="Config with 'sat_encoder'",
        )
        if "required_parameters" in sat_section:
            for param in sat_section["required_parameters"]:
                _check_key(
                    sat_section,
                    param,
                    required=True,
                    expected_type=int,
                    context="sat_encoder",
                )
                if sat_section.get(param, 0) <= 0:
                    raise ValueError(f"sat_encoder: Parameter '{param}' must be positive integer.")

    # NWP Encoders
    _check_key(cfg, "nwp_encoders_dict", required=False, expected_type=dict)

    if nwp_section := cfg.get("nwp_encoders_dict"):
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
                        error_message = (
                            f"{context}: Parameter '{param}' must be positive integer."
                        )
                        raise ValueError(error_message)
        _validate_nwp_specifics(cfg, nwp_section)
