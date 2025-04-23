"""Validation functions for Multimodal configuration"""

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
    raises TypeError.

    Args:
        cfg: The configuration dictionary.
        key: The key to check within the dictionary.
        required: If True, raises KeyError if the key is missing. Defaults to True.
        expected_type (Type | None): The expected type object (e.g., str, int, dict)
            for the key's value. If None, type is not checked. Defaults to None.
        warn_on_type_mismatch: If True and expected_type is provided, logs a
            warning on type mismatch instead of raising TypeError. Defaults to False.
        context: String describing the context for error/warning messages

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
    # Type validation
    if expected_type is not None and not isinstance(value, expected_type):
        message = (
            f"{context} key '{key}' expected type {expected_type.__name__}, "
            f"found {type(value).__name__}."
        )
        # Non critical and critical type mismatches
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

    Args:
        cfg: The main configuration dictionary.
        section_name: The key of the section to validate within `cfg`.
        required: If True, raises KeyError if the section is missing.
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

    # Ensure section follows dict structure
    if not isinstance(section_content, dict):
        raise TypeError(
            f"Config section '{section_name}' must be a dictionary, "
            f"found {type(section_content).__name__}."
        )

    if not section_content and not required:
        logger.warning(f"Optional config section '{section_name}' is present but empty.")
        return section_content

    if check_sub_items_target:
        if not section_content and required:
             raise KeyError(
                 f"Required config section '{section_name}' is empty and requires sub-items "
                 f"with '_target_' keys."
             )
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
    Checking for time params.
    
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

    # Only if associated feature / encoder is configured and enabled
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
    Checks for the presence and structure of required NWP time parameters.

    Checks specifically 'nwp_history_minutes', 'nwp_forecast_minutes') and ensures 
    their keys match sources defined in the 'nwp_encoders_dict'. Logs warnings for 
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

    # Check time param values are integers
    _check_dict_values_are_int(nwp_hist_times, "nwp_history_minutes")
    _check_dict_values_are_int(nwp_forecast_times, "nwp_forecast_minutes")


def _check_output_quantiles_config(cfg: dict[str, Any], context: str = "Top Level") -> None:
    """
    Validates the 'output_quantiles' configuration parameter.
    
    Ensures it exists, is a non-empty list or tuple, and contains only numeric values.

    Args:
        cfg: The configuration dictionary containing 'output_quantiles'.
        context: String describing the context for error messages (e.g., 'Top Level').
                 Defaults to "Top Level".

    Raises:
        KeyError: If 'output_quantiles' key is missing.
        TypeError: If 'output_quantiles' is not a list or tuple, or if any element
                   within it is not an int or float.
        ValueError: If 'output_quantiles' is an empty list or tuple.
    """
    _check_key(cfg, "output_quantiles", required=True, expected_type=(list, tuple), context=context)
    quantiles = cfg["output_quantiles"]
    if not quantiles:
        raise ValueError(f"{context}: 'output_quantiles' list cannot be empty.")
    for i, q_value in enumerate(quantiles):
        if not isinstance(q_value, (int, float)):
             raise TypeError(
                 f"{context}: Element {i} in 'output_quantiles' must be a number, "
                 f"found {type(q_value).__name__}."
            )


def _check_dict_values_are_int(
    data: dict[str, Any], 
    dict_name: str,
) -> None:
    """
    Checks if all values in a dictionary are integers, logs warning otherwise.
    """
    _check_dict_values_type(data, dict_name, int, False)


def _check_convnet_encoder_params(
    cfg: dict[str, Any],
    section_key: str,
    context: str,
    source_key: str | None = None,
) -> None:
    """
    Validates parameters specific to ConvNet encoders.
    """
    convnet_params = [
        "in_channels", "out_features", "number_of_conv3d_layers", 
        "conv3d_channels", "image_size_pixels"
    ]
    _check_encoder_params(cfg, section_key, context, convnet_params, source_key)


def _check_attention_encoder_params(cfg: dict[str, Any], section_key: str, context: str) -> None:
    """
    Validates parameters specific to Attention-based encoders.
    """
    attention_params = [
        "num_sites", "out_features", "num_heads", "kdim", "id_embed_dim"
    ]
    _check_encoder_params(cfg, section_key, context, attention_params, None)


def _get_encoder_config(
    cfg: dict[str, Any],
    section_key: str,
    context: str,
    source_key: str | None = None
) -> dict[str, Any]:
    """
    Retrieves encoder configuration, handling both direct and nested configurations.
    
    Args:
        cfg: The main configuration dictionary.
        section_key: The key within `cfg` that points to the encoder configuration.
        context: Context string for error messages.
        source_key: Optional key for nested configurations.
        
    Returns:
        The encoder configuration dictionary.
    """
    encoder_config: dict[str, Any] | None = None
    if source_key:
        section_dict = cfg.get(section_key)
        if isinstance(section_dict, dict):
            encoder_config = section_dict.get(source_key)
        if encoder_config is None:
            raise KeyError(
                f"{context}: Cannot find valid source config at "
                f"cfg['{section_key}']['{source_key}']."
            )
    else:
        encoder_config = cfg.get(section_key)

    if not isinstance(encoder_config, dict):
        raise TypeError(f"{context}: Encoder configuration must be a dictionary.")
    
    return encoder_config


def _check_positive_int_param(
    config: dict[str, Any],
    param_name: str,
    context: str
) -> None:
    """
    Checks if a parameter exists, is an integer, and is positive.
    
    Args:
        config: The configuration dictionary.
        param_name: The parameter name to check.
        context: Context string for error messages.
    """
    _check_key(config, param_name, required=True, expected_type=int, context=context)
    if config[param_name] <= 0:
        raise ValueError(f"{context}: '{param_name}' must be positive.")


def _check_encoder_params(
    cfg: dict[str, Any],
    section_key: str,
    context: str,
    param_names: list[str],
    source_key: str | None = None
) -> None:
    """
    Generic encoder parameter validation function.
    
    Args:
        cfg: The main configuration dictionary.
        section_key: The key within `cfg` that points to the encoder configuration.
        context: A string describing the context for error messages.
        param_names: List of parameter names to validate as positive integers.
        source_key: An optional key for nested configurations.
    """
    encoder_config = _get_encoder_config(cfg, section_key, context, source_key)
    
    for param_name in param_names:
        _check_positive_int_param(encoder_config, param_name, context)


def _check_dict_values_type(
    data: dict[str, Any],
    dict_name: str,
    expected_type: Type,
    error_on_mismatch: bool = False
) -> None:
    """
    Check if all values in dictionary match expected type.
    
    Args:
        data: The dictionary whose values to check.
        dict_name: Name of the dictionary for error messages.
        expected_type: The expected type for all values.
        error_on_mismatch: If True, raise TypeError on mismatch instead of warning.
    """
    for source, value in data.items():
        if not isinstance(value, expected_type):
            message = (
                f"'{dict_name}' for source '{source}' expected {expected_type.__name__}, "
                f"found {type(value).__name__}."
            )
            if error_on_mismatch:
                raise TypeError(message)
            else:
                logger.warning(message)


def validate_multimodal_config(cfg: dict[str, Any]) -> dict[str, bool]:
    """
    Performs comprehensive validation of Multimodal model configuration dictionary.

    Args:
        cfg: The Multimodal configuration dictionary to validate.

    Returns:
        dict[str, bool]: A dictionary indicating successful validation,
                         typically `{"valid": True}`. If validation fails,
                         an exception is raised instead.

    Raises:
        KeyError: If required top-level keys (e.g., '_target_', 'output_quantiles',
                  'forecast_minutes', 'history_minutes'), required sections
                  (e.g., 'output_network', 'optimizer'), required sub-keys within
                  sections (e.g., '_target_', required time parameters like
                  'sat_history_minutes' when 'sat_encoder' is present), or required
                  encoder parameters are missing.
        TypeError: If configuration elements have incorrect types, such as sections not
                   being dictionaries, 'output_quantiles' not being a list/tuple,
                   or specific parameters within encoders not having the expected
                   numeric types. Note: Non-critical type mismatches for some time
                   parameters might only log warnings.
        ValueError: If certain values are invalid, such as 'output_quantiles' being
                    an empty list, numeric encoder parameters (channels, features,
                    image sizes, etc.) being non-positive, or if keys within NWP
                    time parameter dictionaries ('nwp_history_minutes',
                    'nwp_forecast_minutes') do not exactly match the sources defined
                    in 'nwp_encoders_dict'.
    """
    _check_key(cfg, "_target_", required=True, expected_type=str)
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

    if "nwp_interval_minutes" in cfg and isinstance(cfg.get("nwp_interval_minutes"), dict):
         _check_dict_values_are_int(cfg["nwp_interval_minutes"], "nwp_interval_minutes")

    _check_key(cfg, "embedding_dim", required=False, expected_type=int)
    _check_key(cfg, "include_sun", required=False, expected_type=bool)
    _check_key(cfg, "include_gsp_yield_history", required=False, expected_type=bool)
    _check_key(cfg, "add_image_embedding_channel", required=False, expected_type=bool)

    # Satellite Encoder
    sat_section = _check_dict_section(cfg, "sat_encoder", required=False, check_target=True)

    _check_time_parameter(
        cfg, 
        "sat_history_minutes",
        owner_key="sat_encoder", 
        required_if_owner_present=True,
    )
    
    if sat_section:
        _check_convnet_encoder_params(cfg, section_key="sat_encoder", context="sat_encoder")

    # PV Encoder
    pv_section = _check_dict_section(cfg, "pv_encoder", required=False, check_target=True)

    _check_time_parameter(
        cfg, 
        "pv_history_minutes", 
        owner_key="pv_encoder", 
        required_if_owner_present=True,
    )

    if pv_section:
        _check_attention_encoder_params(cfg, section_key="pv_encoder", context="pv_encoder")

    # NWP Encoders
    nwp_section = _check_dict_section(
        cfg, 
        "nwp_encoders_dict", 
        required=False, 
        check_sub_items_target=True,
    )

    if nwp_section is not None:
        _validate_nwp_specifics(cfg, nwp_section)
        for source in nwp_section.keys():
             _check_convnet_encoder_params(
                cfg=cfg,
                section_key="nwp_encoders_dict",
                context=f"nwp_encoders_dict[{source}]",
                source_key=source
            )

    logger.info("Multimodal configuration validation successful.")
    return {"valid": True}
