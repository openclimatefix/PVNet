import logging


logger = logging.getLogger(__name__)


def _check_key(
    cfg,
    key,
    required = True,
    expected_type = None,
    warn_on_type_mismatch = False,
    context = "Configuration"
):
    """
    Checks for presence and optionally type of a key in the config.
    Logs a warning on type mismatch if warn_on_type_mismatch is True.

    Args:
        cfg: The configuration dictionary.
        key: The key to check.
        required: If True, raises KeyError if the key is missing.
        expected_type: If provided, checks if the key's value is of this type.
        warn_on_type_mismatch: If True and expected_type is provided, logs a
            warning on type mismatch instead of raising TypeError.
        context: String describing the context for error/warning messages.

    Raises:
        KeyError: If required is True and the key is missing.
        TypeError: If expected_type is provided, the type mismatches, and
                   warn_on_type_mismatch is False.
    """
    if key not in cfg:
        if required:
            raise KeyError(f"{context} missing required key: '{key}'")
        else:
            return

    value = cfg[key]
    if expected_type is not None and not isinstance(value, expected_type):
        message = (f"{context} key '{key}' expected type {expected_type.__name__}, "
                   f"found {type(value).__name__}.")
        if warn_on_type_mismatch:
            logger.warning(message)
        else:
            raise TypeError(message)


def _check_dict_section(
    cfg,
    section_name,
    required = True,
    check_target = True,
    check_sub_items_target = False
):
    """
    Validates a section expected to be a dictionary, optionally checking for '_target_'.
    Logs a warning if an optional section is present but empty.

    Args:
        cfg: The main configuration dictionary.
        section_name: The key of the section to validate.
        required: If True, raises KeyError if the section is missing.
        check_target: If True, checks for '_target_' key within the section dict.
        check_sub_items_target: If True, assumes the section dict contains further
                                dicts and checks for '_target_' within those sub-dicts.

    Returns:
        The section dictionary if validation passes, or None if optional and absent/empty.

    Raises:
        KeyError: If required and section/sub-key is missing.
        TypeError: If the section or its sub-items are not dictionaries as expected.
    """
    if section_name not in cfg:
        if required:
            raise KeyError(f"Configuration missing required section: '{section_name}'")
        else:
            return None

    section_content = cfg.get(section_name)

    if not isinstance(section_content, dict):
        raise TypeError(f"Config section '{section_name}' must be a dictionary, "
                        f"found {type(section_content)}.")

    if not section_content:
        if not required:
             logger.warning(f"Optional config section '{section_name}' is present but empty.")
        elif required and check_target:
             raise KeyError(f"Required config section '{section_name}' is empty and requires sub-key: '_target_'")
        return section_content


    if check_sub_items_target:
        for source, sub_config in section_content.items():
            if not isinstance(sub_config, dict):
                raise TypeError(f"Config for source '{source}' in '{section_name}' "
                                f"must be a dictionary.")
            if "_target_" not in sub_config:
                raise KeyError(f"Source '{source}' in section '{section_name}' "
                               f"missing required sub-key: '_target_'")
    elif check_target:
        if "_target_" not in section_content:
            raise KeyError(f"Config section '{section_name}' is missing required "
                           f"sub-key: '_target_'")

    return section_content


def _check_time_parameter(
    cfg,
    param_name,
    owner_key,
    required_if_owner_present = True
):
    """
    Checks a time parameter associated with an optional feature/encoder.
    Logs a warning if required_if_owner_present is False and the parameter is missing.

    Args:
        cfg: The main configuration dictionary.
        param_name: The name of the time parameter key (e.g., 'sat_history_minutes').
        owner_key: The key indicating the feature is enabled (e.g., 'sat_encoder').
        required_if_owner_present: If True, raise KeyError if the owner_key is
                                  present/truthy but the param_name is missing.
                                  If False, log a warning instead.

    Raises:
        KeyError: If the owner is present and required_if_owner_present is True
                  and the time parameter is missing.
    """
    if owner_key in cfg and cfg.get(owner_key):
        context = f"Config includes '{owner_key}'"
        if param_name not in cfg:
            message = f"{context} but is missing '{param_name}'."
            if required_if_owner_present:
                raise KeyError(message)
            else:
                logger.warning(f"{message} (may use default).")
        else:
            _check_key(cfg, param_name, required=False, expected_type=int,
                       warn_on_type_mismatch=True, context=f"Parameter '{param_name}'")


def _validate_nwp_specifics(cfg, nwp_section):
    """
    Performs validation specific to the NWP configuration.
    Logs warnings for empty NWP source list or non-integer time values.

    Args:
        cfg: The main configuration dictionary.
        nwp_section: The validated 'nwp_encoders_dict' section content.

    Raises:
        KeyError: If required NWP time parameter keys are missing.
        TypeError: If NWP time parameters are not dictionaries.
        ValueError: If keys in NWP time dicts don't match nwp_encoders_dict keys.
    """
    nwp_sources = list(nwp_section.keys())
    if not nwp_sources:
        logger.warning("'nwp_encoders_dict' is defined but contains no NWP sources.")
        return

    context = "Config includes 'nwp_encoders_dict' with sources"
    _check_key(cfg, "nwp_history_minutes", required=True, expected_type=dict, context=context)
    _check_key(cfg, "nwp_forecast_minutes", required=True, expected_type=dict, context=context)

    nwp_hist_times = cfg["nwp_history_minutes"]
    nwp_forecast_times = cfg["nwp_forecast_minutes"]

    hist_keys = set(nwp_hist_times.keys())
    forecast_keys = set(nwp_forecast_times.keys())
    encoder_keys = set(nwp_sources)

    # Raise errors for key mismatches
    if hist_keys != encoder_keys:
        raise ValueError(f"Keys in 'nwp_history_minutes' {hist_keys} do not match "
                         f"sources in 'nwp_encoders_dict' {encoder_keys}.")
    if forecast_keys != encoder_keys:
        raise ValueError(f"Keys in 'nwp_forecast_minutes' {forecast_keys} do not match "
                         f"sources in 'nwp_encoders_dict' {encoder_keys}.")

    # Log warnings for time values
    for source, value in nwp_hist_times.items():
        if not isinstance(value, int):
            logger.warning(f"'nwp_history_minutes' for source '{source}' expected int, "
                           f"found {type(value).__name__}.")
    for source, value in nwp_forecast_times.items():
        if not isinstance(value, int):
            logger.warning(f"'nwp_forecast_minutes' for source '{source}' expected int, "
                           f"found {type(value).__name__}.")


def validate_multimodal_config(cfg):
    """
    Validates the configuration dictionary for the Multimodal Model using helpers.
    Logs warnings for non-critical issues detected during validation.

    Checks that the configuration has the expected structure and keys based
    on the Multimodal Model requirements. Critical issues (missing required keys,
    incorrect types for sections, mismatched NWP keys) will raise an exception.

    Args:
        cfg: The configuration dictionary to validate.

    Returns:
        dict: A dictionary indicating success: `{"valid": True}`.
              If validation fails due to a critical issue, an exception is raised.

    Raises:
        TypeError: If a configuration section expected to be a dictionary is not,
                   or if required time parameters (e.g., for NWP) are not dictionaries.
        KeyError: For critical validation failures like missing required keys
                  (e.g., '_target_', 'forecast_minutes', 'output_network',
                  'optimizer', required sub-keys like '_target_').
        ValueError: If keys in NWP time parameter dicts do not match the keys
                    in the nwp_encoders_dict.
    """

    _check_key(cfg, "_target_", required=True, expected_type=str)
    _check_key(cfg, "forecast_minutes", required=True, expected_type=int, warn_on_type_mismatch=True)
    _check_key(cfg, "history_minutes", required=True, expected_type=int, warn_on_type_mismatch=True)

    _check_dict_section(cfg, "output_network", required=True, check_target=True)
    _check_dict_section(cfg, "optimizer", required=True, check_target=True)

    nwp_section = _check_dict_section(cfg, "nwp_encoders_dict", required=False, check_sub_items_target=True)

    _check_dict_section(cfg, "sat_encoder", required=False, check_target=True)
    _check_dict_section(cfg, "pv_encoder", required=False, check_target=True)
    _check_dict_section(cfg, "sensor_encoder", required=False, check_target=True)

    _check_time_parameter(cfg, "sat_history_minutes", owner_key="sat_encoder", required_if_owner_present=True)
    _check_time_parameter(cfg, "pv_history_minutes", owner_key="pv_encoder", required_if_owner_present=True)
    _check_time_parameter(cfg, "sensor_history_minutes", owner_key="sensor_encoder", required_if_owner_present=False)
    _check_time_parameter(cfg, "sensor_forecast_minutes", owner_key="sensor_encoder", required_if_owner_present=False)

    if nwp_section:
         _validate_nwp_specifics(cfg, nwp_section)

    return {"valid": True}
