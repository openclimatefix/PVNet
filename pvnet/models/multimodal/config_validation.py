"""Validation functions for Multimodal configuration"""

import logging
from typing import Any, Type

import numpy as np
from ocf_data_sampler.torch_datasets.sample.base import NumpyBatch

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

    Raises:
        KeyError: If `required` is True and the `key` is missing.
        TypeError: If value type mismatches `expected_type` and `warn_on_type_mismatch` is False.
    """
    if key not in cfg:
        if required:
            raise KeyError(f"{context} missing required key: '{key}'")
        else:
            return

    value: Any = cfg[key]
    # Type validation checking
    if expected_type is not None and not isinstance(value, expected_type):
        message = (
            f"{context} key '{key}' expected type {expected_type.__name__}, "
            f"found {type(value).__name__}."
        )
        # Non critical / critical type mismatches
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
    """Check for a time parameter if its associated feature key is present/truthy.

    Args:
        cfg: The main configuration dictionary.
        param_name: Name of the time parameter key (e.g., 'sat_history_minutes').
        owner_key: Key indicating the associated feature is enabled (e.g., 'sat_encoder').
        required_if_owner_present: If True, raise error if param_name is missing.

    Raises:
        KeyError: If `required_if_owner_present` is True and the parameter is missing.
    """

    # Only if associated feature / encoder is configured and enabled
    if owner_key in cfg and cfg.get(owner_key):
        context = f"Config includes '{owner_key}'"
        if param_name not in cfg:
            message = f"{context} but is missing '{param_name}'."
            if required_if_owner_present:
                raise KeyError(message)
            else:
                logger.warning(
                    f"{message} (Action: May use default value if applicable)."
                )
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
    nwp_intervals: dict[str, Any] = cfg["nwp_interval_minutes"]
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
    _check_dict_values_are_int(nwp_hist_times, "nwp_history_minutes")
    _check_dict_values_are_int(nwp_forecast_times, "nwp_forecast_minutes")
    _check_dict_values_are_int(nwp_intervals, "nwp_interval_minutes")


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
    if not quantiles:
        raise ValueError(f"{context}: 'output_quantiles' list cannot be empty.")
    for i, q_value in enumerate(quantiles):
        if not isinstance(q_value, (int, float)):
            raise TypeError(
                f"{context}: Element {i} in 'output_quantiles' must be a number, "
                f"found {type(q_value).__name__}."
            )


def _check_convnet_encoder_params(
    cfg: dict[str, Any], section_key: str, context: str, source_key: str | None = None
) -> None:
    """Validate required positive integer parameters for ConvNet encoders.

    Args:
        cfg: The main configuration dictionary.
        section_key: Key of the encoder section (e.g., 'sat_encoder', 'nwp_encoders_dict').
        context: Context for error messages.
        source_key: Sub-key if section_key points to a dict of encoders (e.g., NWP source).

    Raises:
        KeyError, TypeError, ValueError: If parameters are missing, wrong type, or not positive.
    """
    convnet_params = [
        "in_channels",
        "out_features",
        "number_of_conv3d_layers",
        "conv3d_channels",
        "image_size_pixels",
    ]
    encoder_config = _get_encoder_config(cfg, section_key, context, source_key)
    _check_positive_int_params_in_dict(encoder_config, convnet_params, context)


def _check_attention_encoder_params(
    cfg: dict[str, Any], section_key: str, context: str
) -> None:
    """Validate required positive integer parameters for Attention-based encoders.

    Args:
        cfg: The main configuration dictionary.
        section_key: Key of the encoder section (e.g., 'pv_encoder').
        context: Context for error messages.

    Raises:
        KeyError, TypeError, ValueError: If parameters are missing, wrong type, or not positive.
    """
    attention_params = [
        "num_sites",
        "out_features",
        "num_heads",
        "kdim",
        "id_embed_dim",
    ]
    encoder_config = _get_encoder_config(cfg, section_key, context, None)
    _check_positive_int_params_in_dict(encoder_config, attention_params, context)


def _get_time_steps(hist_mins: int, forecast_mins: int, interval: int) -> tuple:
    """Calculate history and forecast time steps based on duration and interval.

    Args:
        hist_mins: Duration of history in minutes.
        forecast_mins: Duration of forecast in minutes.
        interval: Time interval between steps in minutes.

    Returns:
        Tuple containing (history_steps, forecast_steps). Includes t0 for history.

    Raises:
        ValueError: If interval is not positive.
    """
    if interval <= 0:
        raise ValueError("Time interval must be positive")
    hist_steps = int(hist_mins) // interval + 1
    forecast_steps = int(forecast_mins) // interval
    return hist_steps, forecast_steps


def _check_batch_size_consistency(
    current_batch_size: int | None, arr: np.ndarray, name: str
) -> int:
    """Check if an array's first dimension matches the expected batch size.

    Args:
        current_batch_size: The batch size inferred so far (or None).
        arr: The numpy array to check.
        name: Name of the array/modality for error messages.

    Returns:
        The consistent batch size (inferred or validated).

    Raises:
        ValueError: If batch size is inconsistent or array batch dimension is <= 0.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"{name} data must be a NumPy array, found {type(arr).__name__}"
        )
    if arr.ndim == 0:
        raise ValueError(f"{name} array is scalar, cannot determine batch size.")

    actual_size = arr.shape[0]
    if actual_size <= 0:
        raise ValueError(f"{name} batch dimension has size <= 0: {actual_size}")

    if current_batch_size is None:
        logger.debug(f"Inferred batch size: {actual_size} from {name}")
        return actual_size
    elif current_batch_size != actual_size:
        raise ValueError(
            f"Batch size mismatch for {name}: expected {current_batch_size}, got {actual_size}"
        )
    else:
        # Batch size matches the existing one
        return current_batch_size


def _get_modality_interval(
    cfg: dict, modality_key_base: str, default_interval_minutes: int
) -> int:
    """Get time interval (minutes) for a modality, using config or default.

    Args:
        cfg: The main configuration dictionary.
        modality_key_base: Base key for the modality (e.g., 'satellite_actual', 'pv', 'sun').
        default_interval_minutes: Default interval if specific key isn't found/valid.

    Returns:
        The determined interval in minutes (positive integer).
    """
    config_key = f"{modality_key_base}_time_resolution_minutes"
    interval = cfg.get(config_key)

    if interval is not None:
        if isinstance(interval, int) and interval > 0:
            logger.debug(f"Using interval from config key '{config_key}': {interval}")
            return interval
        else:
            logger.warning(
                f"Config key '{config_key}' found but is not a positive integer ({interval}). "
                f"Falling back to default interval {default_interval_minutes}."
            )

    logger.debug(
        f"Using default interval for {modality_key_base}: {default_interval_minutes}"
    )
    return default_interval_minutes


def _check_dict_values_are_int(
    data: dict[str, Any],
    dict_name: str,
) -> None:
    """Check if all values in a dictionary are integers (logs warning on mismatch).

    Args:
        data: The dictionary whose values to check.
        dict_name: Name of the dictionary for warning messages.
    """
    _check_dict_values_type(data, dict_name, int, error_on_mismatch=False)


def _check_dict_values_type(
    data: dict[str, Any],
    dict_name: str,
    expected_type: Type,
    error_on_mismatch: bool = False,
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


def _get_encoder_config(
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
                "found {type(section_dict).__name__}."
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
        raise TypeError(
            f"{context}: Final resolved encoder config is not a dictionary."
        )

    return encoder_config


def _check_positive_int_params_in_dict(
    config_dict: dict[str, Any], param_names: list[str], context: str
) -> None:
    """Check if specified keys in a dict exist, are integers, and are positive.

    Args:
        config_dict: The dictionary containing the parameters.
        param_names: A list of keys to check.
        context: Context string for error messages.

    Raises:
        KeyError: If a parameter key is missing.
        TypeError: If a parameter value is not an integer.
        ValueError: If a parameter value is not positive.
    """
    for param_name in param_names:
        # Check presence and type first
        _check_key(
            config_dict, param_name, required=True, expected_type=int, context=context
        )
        # If type check passes, check value
        if config_dict[param_name] <= 0:
            raise ValueError(
                f"{context}: Parameter '{param_name}' must be a positive integer, "
                "found {config_dict[param_name]}."
            )


def _validate_static_config(cfg: dict[str, Any]) -> None:
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

    # Check required sections are dicts with targets
    _check_dict_section(cfg, "output_network", required=True, check_target=True)
    _check_dict_section(cfg, "optimizer", required=True, check_target=True)

    # Satellite Encoder
    sat_section = _check_dict_section(
        cfg, "sat_encoder", required=False, check_target=True
    )
    _check_time_parameter(
        cfg,
        "sat_history_minutes",
        owner_key="sat_encoder",
        required_if_owner_present=True,
    )
    if sat_section:
        _check_convnet_encoder_params(
            cfg, section_key="sat_encoder", context="sat_encoder"
        )

    # PV Encoder
    pv_section = _check_dict_section(
        cfg, "pv_encoder", required=False, check_target=True
    )
    _check_time_parameter(
        cfg,
        "pv_history_minutes",
        owner_key="pv_encoder",
        required_if_owner_present=True,
    )
    if pv_section:
        _check_attention_encoder_params(
            cfg, section_key="pv_encoder", context="pv_encoder"
        )

    # NWP Encoders
    nwp_section = _check_dict_section(
        cfg, "nwp_encoders_dict", required=False, check_sub_items_target=True
    )
    if nwp_section is not None:
        _validate_nwp_specifics(cfg, nwp_section)
        for source in nwp_section.keys():
            _check_convnet_encoder_params(
                cfg=cfg,
                section_key="nwp_encoders_dict",
                context=f"nwp_encoders_dict[{source}]",
                source_key=source,
            )


def validate(
    numpy_batch: NumpyBatch,
    multimodal_config: dict,
    default_interval_minutes: int = 5,
) -> None:
    """Validate a batch of numpy data against the multimodal configuration.

    Performs static config validation first, then checks shapes of data arrays
    in the batch against expectations derived from the config.

    Args:
        numpy_batch: Dictionary containing modality data as NumPy arrays.
        multimodal_config: The multimodal configuration dictionary.
        default_interval_minutes: Default time resolution (minutes) if not in config.

    Raises:
        KeyError, TypeError, ValueError: If config is invalid or batch shapes mismatch config.
    """

    # Check config structure primarily
    try:
        _validate_static_config(multimodal_config)
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    logger.info("Validating batch data shapes against configuration")
    cfg = multimodal_config
    batch_size: int | None = None

    # Satellite validation
    if "sat_encoder" in cfg and cfg.get("sat_encoder"):
        key = "satellite_actual"
        if key not in numpy_batch:
            raise KeyError(f"Batch missing '{key}' data")
        data = numpy_batch[key]
        if not isinstance(data, np.ndarray):
            raise TypeError(f"'{key}' data must be np.ndarray")

        interval = _get_modality_interval(cfg, key, default_interval_minutes)

        sat_cfg = _get_encoder_config(cfg, "sat_encoder", "sat_encoder")
        h = w = sat_cfg["image_size_pixels"]
        c = sat_cfg["in_channels"]
        hist_steps, _ = _get_time_steps(cfg["sat_history_minutes"], 0, interval)
        expected_shape = (hist_steps, c, h, w)

        if data.ndim != 5 or data.shape[1:] != expected_shape:
            raise ValueError(
                f"'{key}' shape error using interval {interval}. "
                "Expected B x {expected_shape}, Got {data.shape}"
            )
        batch_size = _check_batch_size_consistency(batch_size, data, key)

    # NWP validation
    if "nwp_encoders_dict" in cfg and cfg.get("nwp_encoders_dict"):
        key = "nwp"
        if key not in numpy_batch:
            raise KeyError(f"Batch missing '{key}' data dict")
        nwp_batch_data = numpy_batch[key]
        if not isinstance(nwp_batch_data, dict):
            raise TypeError(f"'{key}' data must be a dict")

        nwp_cfg_dict = cfg["nwp_encoders_dict"]
        config_sources = set(nwp_cfg_dict.keys())
        batch_sources = set(nwp_batch_data.keys())

        if not config_sources.issubset(batch_sources):
            missing_in_batch = config_sources - batch_sources
            raise KeyError(
                f"NWP data in batch is missing configured sources: {missing_in_batch}"
            )

        extra_in_batch = batch_sources - config_sources
        if extra_in_batch:
            logger.warning(
                f"NWP data in batch contains extra sources not in config: {extra_in_batch}. "
                "These will be ignored by validation."
            )

        nwp_hist_mins = cfg["nwp_history_minutes"]
        nwp_forecast_mins = cfg["nwp_forecast_minutes"]
        nwp_intervals = cfg["nwp_interval_minutes"]

        for source in config_sources:
            if source not in nwp_batch_data or not isinstance(
                nwp_batch_data[source], dict
            ):
                raise TypeError(
                    f"NWP data for configured source '{source}' is missing or not a dict in batch."
                )
            source_nwp_dict = nwp_batch_data[source]

            try:
                source_data_array = next(
                    v
                    for v in source_nwp_dict.values()
                    if isinstance(v, np.ndarray) and v.ndim > 0
                )
            except StopIteration:
                raise TypeError(
                    f"NWP source '{source}' dict in batch contains no suitable "
                    "numpy array for validation."
                )

            source_cfg = _get_encoder_config(
                cfg, "nwp_encoders_dict", f"nwp[{source}]", source_key=source
            )
            h = w = source_cfg["image_size_pixels"]
            c = source_cfg["in_channels"]
            interval = nwp_intervals[source]
            if not isinstance(interval, int) or interval <= 0:
                raise ValueError(
                    f"NWP interval for source '{source}' must be a positive integer "
                    "in config, got {interval}."
                )
            hist_steps, forecast_steps = _get_time_steps(
                nwp_hist_mins[source], nwp_forecast_mins[source], interval
            )
            expected_shape = (hist_steps + forecast_steps, c, h, w)

            if (
                source_data_array.ndim != 5
                or source_data_array.shape[1:] != expected_shape
            ):
                raise ValueError(
                    f"NWP source '{source}' shape error using interval {interval}. "
                    "Expected B x {expected_shape}, Got {source_data_array.shape}"
                )

            batch_size = _check_batch_size_consistency(
                batch_size, source_data_array, f"{key}[{source}]"
            )

    # PV validation
    if "pv_encoder" in cfg and cfg.get("pv_encoder"):
        key = "pv"
        if key not in numpy_batch:
            raise KeyError(f"Batch missing '{key}' data")
        data = numpy_batch[key]
        if not isinstance(data, np.ndarray):
            raise TypeError(f"'{key}' data must be np.ndarray")

        interval = _get_modality_interval(cfg, key, default_interval_minutes)

        pv_cfg = _get_encoder_config(cfg, "pv_encoder", "pv_encoder")
        num_sites = pv_cfg["num_sites"]
        hist_steps, _ = _get_time_steps(cfg["pv_history_minutes"], 0, interval)
        expected_shape = (hist_steps, num_sites)

        if data.ndim != 3 or data.shape[1:] != expected_shape:
            raise ValueError(
                f"'{key}' shape error using interval {interval}. "
                "Expected B x {expected_shape}, Got {data.shape}"
            )
        batch_size = _check_batch_size_consistency(batch_size, data, key)

    # GSP yield history inclusion
    if cfg.get("include_gsp_yield_history"):
        key = "gsp"
        if key not in numpy_batch:
            raise KeyError(f"Batch missing '{key}' data")
        data = numpy_batch[key]
        if not isinstance(data, np.ndarray):
            raise TypeError(f"'{key}' data must be np.ndarray")

        interval = _get_modality_interval(cfg, key, default_interval_minutes)

        hist_steps, forecast_steps = _get_time_steps(
            cfg["history_minutes"], cfg["forecast_minutes"], interval
        )
        expected_time_steps = hist_steps + forecast_steps
        if data.ndim < 2 or data.ndim > 3 or data.shape[1] != expected_time_steps:
            raise ValueError(
                f"'{key}' shape error using interval {interval}. "
                "Expected B x ({expected_time_steps}, [1]), Got {data.shape}"
            )
        if data.ndim == 3 and data.shape[2] != 1:
            logger.warning(f"'{key}' has > 1 feature ({data.shape[2]}), expected 1.")
        batch_size = _check_batch_size_consistency(batch_size, data, key)

    # Solar inclusion
    if cfg.get("include_sun"):
        sun_interval = _get_modality_interval(cfg, "sun", default_interval_minutes)
        hist_steps, forecast_steps = _get_time_steps(
            cfg["history_minutes"], cfg["forecast_minutes"], sun_interval
        )
        expected_time_steps = hist_steps + forecast_steps

        key_az = "solar_azimuth"
        if key_az not in numpy_batch:
            raise KeyError(
                f"Batch missing '{key_az}' data (required since include_sun=True)"
            )
        data_az = numpy_batch[key_az]
        if not isinstance(data_az, np.ndarray):
            raise TypeError(f"'{key_az}' data must be np.ndarray")
        if data_az.ndim != 2 or data_az.shape[1] != expected_time_steps:
            raise ValueError(
                f"'{key_az}' shape error using interval {sun_interval}. "
                "Expected B x ({expected_time_steps}), Got {data_az.shape}"
            )
        batch_size = _check_batch_size_consistency(batch_size, data_az, key_az)

        key_el = "solar_elevation"
        if key_el not in numpy_batch:
            raise KeyError(
                f"Batch missing '{key_el}' data (required since include_sun=True)"
            )
        data_el = numpy_batch[key_el]
        if not isinstance(data_el, np.ndarray):
            raise TypeError(f"'{key_el}' data must be np.ndarray")
        if data_el.ndim != 2 or data_el.shape[1] != expected_time_steps:
            raise ValueError(
                f"'{key_el}' shape error using interval {sun_interval}. "
                "Expected B x ({expected_time_steps}), Got {data_el.shape}"
            )
        batch_size = _check_batch_size_consistency(batch_size, data_el, key_el)

    if batch_size is None:
        logger.warning(
            "Batch size could not be determined (no modalities checked or empty batch?)."
        )

    logger.info("Batch data shape validation successful against configuration.")
