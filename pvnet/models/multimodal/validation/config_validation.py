"""Main validation function for multimodal configuration and batches."""

import logging

import numpy as np
from ocf_data_sampler.torch_datasets.sample.base import NumpyBatch
from omegaconf import DictConfig, OmegaConf

from pvnet.models.multimodal.validation.config_batch_validation import (
    check_batch_data,
    get_modality_interval,
    get_time_steps,
    validate_array_shape,
    validate_nwp_source_structure,
)
from pvnet.models.multimodal.validation.config_static_validation import (
    get_encoder_config,
    validate_static_config,
)

logger = logging.getLogger(__name__)


def _validate_satellite_data(
    numpy_batch: NumpyBatch,
    mm_config_dict: dict,
    input_data_dict: dict,
    expected_batch_size: int,
):
    config_key_sat = "sat_encoder"
    data_key_sat = "satellite_actual"
    logger.debug(f"Validating modality: {data_key_sat}")

    data = check_batch_data(numpy_batch, data_key_sat, np.ndarray, config_key_sat)
    time_res_mins = get_modality_interval(
        input_data_dict,
        "satellite",
        secondary_modality_key=None
    )
    sat_cfg = get_encoder_config(mm_config_dict, config_key_sat, config_key_sat)
    h = w = sat_cfg["image_size_pixels"]
    c = sat_cfg["in_channels"]
    hist_mins = mm_config_dict["sat_history_minutes"]
    hist_steps, _ = get_time_steps(hist_mins, 0, time_res_mins)

    expected_shape_no_batch = (hist_steps, c, h, w)
    full_expected_shape = (expected_batch_size,) + expected_shape_no_batch
    dimension_names = ["batch_size", "time_steps", "channels", "height", "width"]

    validate_array_shape(
        data=data,
        expected_shape_with_batch=full_expected_shape,
        data_key=data_key_sat,
        time_resolution_minutes=time_res_mins,
        dim_names=dimension_names
    )
    logger.debug(
        f"'{data_key_sat}' shape validation passed "
        f"(time resolution: {time_res_mins} mins)."
    )


def _validate_nwp_data(
    numpy_batch: NumpyBatch,
    mm_config_dict: dict,
    input_data_dict: dict,
    expected_batch_size: int,
):
    config_key_nwp = "nwp_encoders_dict"
    data_key_nwp = "nwp"
    logger.debug(f"Validating modality: {data_key_nwp}")

    nwp_batch_data = check_batch_data(numpy_batch, data_key_nwp, dict, config_key_nwp)
    nwp_model_cfg_dict = mm_config_dict[config_key_nwp]
    config_sources = set(nwp_model_cfg_dict.keys())
    batch_sources = set(nwp_batch_data.keys())

    if not config_sources.issubset(batch_sources):
        raise KeyError(f"NWP batch missing configured sources: {config_sources - batch_sources}")
    if batch_sources - config_sources:
        extra_sources = batch_sources - config_sources
        logger.warning(f"NWP batch has extra sources not in config: {extra_sources}")

    nwp_hist_mins_dict = mm_config_dict["nwp_history_minutes"]
    nwp_forecast_mins_dict = mm_config_dict["nwp_forecast_minutes"]
    nwp_interval_mins_config_dict = mm_config_dict.get("nwp_interval_minutes", {})

    for source in config_sources:
        source_key_str = f"{data_key_nwp}[{source}]"
        logger.debug(f"Validating NWP source: {source}")
        source_data_array = validate_nwp_source_structure(nwp_batch_data, source)

        time_res_mins = nwp_interval_mins_config_dict.get(source)
        if time_res_mins is None:
            logger.warning(
                f"Time resolution for NWP source '{source}' not found in "
                f"mm_config_dict['nwp_interval_minutes']. Defaulting to 60 minutes for validation."
            )
            time_res_mins = 60

        source_model_cfg = get_encoder_config(
            mm_config_dict, config_key_nwp, source_key_str, source_key=source
        )
        h = w = source_model_cfg["image_size_pixels"]
        c = source_model_cfg["in_channels"]
        hist_mins_source = nwp_hist_mins_dict.get(source)
        forecast_mins_source = nwp_forecast_mins_dict.get(source)

        if hist_mins_source is None:
            raise KeyError(
                f"History minutes missing for NWP source '{source}' "
                f"in mm_config_dict['nwp_history_minutes']."
            )

        if forecast_mins_source is None:
            raise KeyError(
                f"Forecast minutes missing for NWP source '{source}' "
                f"in mm_config_dict['nwp_forecast_minutes']."
            )

        hist_steps, forecast_steps = get_time_steps(
            hist_mins_source, forecast_mins_source, time_res_mins
        )
        expected_shape_no_batch = (hist_steps + forecast_steps, c, h, w)
        full_expected_shape = (expected_batch_size,) + expected_shape_no_batch
        dimension_names = ["batch_size", "time_steps", "channels", "height", "width"]

        validate_array_shape(
            data=source_data_array,
            expected_shape_with_batch=full_expected_shape,
            data_key=source_key_str,
            time_resolution_minutes=time_res_mins,
            dim_names=dimension_names
        )
        logger.debug(
            f"NWP source '{source}' shape validation passed "
            f"(time resolution: {time_res_mins} mins)."
        )


def _validate_pv_data(
    numpy_batch: NumpyBatch,
    mm_config_dict: dict,
    input_data_dict: dict,
    expected_batch_size: int,
):
    config_key_pv = "pv_encoder"
    data_key_pv = "pv"

    possible_keys = ["site", "pv"]
    modality_data_cfg_key_for_interval = next(
        (k_pv for k_pv in possible_keys if k_pv in input_data_dict and
            isinstance(input_data_dict.get(k_pv), dict)),
        None
    )
    if modality_data_cfg_key_for_interval is None:
            raise KeyError(
                "Neither 'site' nor 'pv' section found as a dictionary in "
                "input_data_dict for PV validation"
            )

    logger.debug(
        f"Validating modality: {data_key_pv} using input config "
        f"'{modality_data_cfg_key_for_interval}'"
    )
    data = check_batch_data(numpy_batch, data_key_pv, np.ndarray, config_key_pv)
    time_res_mins = get_modality_interval(input_data_dict, modality_data_cfg_key_for_interval, None)

    pv_cfg = get_encoder_config(mm_config_dict, config_key_pv, config_key_pv)
    num_sites = pv_cfg["num_sites"]
    hist_mins = mm_config_dict["pv_history_minutes"]
    hist_steps, _ = get_time_steps(hist_mins, 0, time_res_mins)

    expected_shape_no_batch = (hist_steps, num_sites)
    full_expected_shape = (expected_batch_size,) + expected_shape_no_batch
    dimension_names = ["batch_size", "time_steps", "num_sites"]

    validate_array_shape(
        data=data,
        expected_shape_with_batch=full_expected_shape,
        data_key=data_key_pv,
        time_resolution_minutes=time_res_mins,
        allow_ndim_plus_one=True,
        dim_names=dimension_names
    )
    logger.debug(
        f"'{data_key_pv}' shape validation passed "
        f"(time resolution: {time_res_mins} mins)."
    )


def _validate_gsp_data(
    numpy_batch: NumpyBatch,
    mm_config_dict: dict,
    input_data_dict: dict,
    expected_batch_size: int,
):
    config_key_gsp = "include_gsp_yield_history"
    data_key_gsp = "gsp"
    logger.debug(f"Validating modality: {data_key_gsp}")

    data = check_batch_data(numpy_batch, data_key_gsp, np.ndarray, config_key_gsp)
    time_res_mins = get_modality_interval(input_data_dict, "gsp", None)

    hist_steps, forecast_steps = get_time_steps(
        mm_config_dict["history_minutes"], mm_config_dict["forecast_minutes"], time_res_mins
    )
    expected_shape_no_batch = (hist_steps + forecast_steps,)
    full_expected_shape = (expected_batch_size,) + expected_shape_no_batch
    dimension_names = ["batch_size", "time_steps"]

    validate_array_shape(
        data=data,
        expected_shape_with_batch=full_expected_shape,
        data_key=data_key_gsp,
        time_resolution_minutes=time_res_mins,
        allow_ndim_plus_one=True,
        dim_names=dimension_names
    )
    logger.debug(
        f"'{data_key_gsp}' shape validation passed "
        f"(time resolution: {time_res_mins} mins)."
    )


def _validate_sun_data(
    numpy_batch: NumpyBatch,
    mm_config_dict: dict,
    input_data_dict: dict,
    expected_batch_size: int,
):
    config_key_sun = "include_sun"
    logger.debug("Validating modality: sun (azimuth, elevation)")

    possible_fallback_keys = ["gsp", "site", "pv"]
    fallback_key_sun = next(
        (k_sun for k_sun in possible_fallback_keys if k_sun in input_data_dict and
            isinstance(input_data_dict.get(k_sun), dict)),
        None
    )
    try:
        sun_time_res_mins = get_modality_interval(input_data_dict, "sun", fallback_key_sun)
        log_message_sun_res = f"Time resolution for 'sun' obtained: {sun_time_res_mins} mins."
        has_direct_sun_time_res = (
            input_data_dict.get("sun") and
            isinstance(input_data_dict.get("sun"), dict) and
            'time_resolution_minutes' in input_data_dict.get("sun", {})
        )
        if fallback_key_sun and not has_direct_sun_time_res:
            log_message_sun_res += f" (Used fallback: '{fallback_key_sun}')"
        elif has_direct_sun_time_res:
            log_message_sun_res += " (Used direct 'sun' config)"
        logger.debug(log_message_sun_res)

    except (KeyError, ValueError) as e:
        if (fallback_key_sun is None and
                not (
                    input_data_dict.get("sun") and
                    isinstance(input_data_dict.get("sun"), dict)
                ) and
                "modality 'sun'" in str(e) and
                "fallback" not in str(e).lower()):
            raise KeyError(
                    "Cannot determine time resolution for sun: No direct 'sun' config with "
                    "'time_resolution_minutes' found, and no suitable fallback modality ('gsp', "
                    "'site', or 'pv') with a dictionary structure was found in input_data_dict."
            ) from e
        raise KeyError(f"Failed to determine time resolution for 'sun': {e}") from e

    hist_steps, forecast_steps = get_time_steps(
        mm_config_dict["history_minutes"], mm_config_dict["forecast_minutes"], sun_time_res_mins
    )
    expected_shape_no_batch = (hist_steps + forecast_steps,)
    full_expected_shape = (expected_batch_size,) + expected_shape_no_batch
    dimension_names = ["batch_size", "time_steps"]

    for key_sun_data in ["solar_azimuth", "solar_elevation"]:
        data_sun = check_batch_data(numpy_batch, key_sun_data, np.ndarray, config_key_sun)
        validate_array_shape(
            data=data_sun,
            expected_shape_with_batch=full_expected_shape,
            data_key=key_sun_data,
            time_resolution_minutes=sun_time_res_mins,
            dim_names=dimension_names
        )
    logger.debug(f"'sun' data shape validation passed (time resolution: {sun_time_res_mins} mins).")


def validate(
    numpy_batch: NumpyBatch,
    multimodal_config: DictConfig,
    input_data_config: DictConfig,
    expected_batch_size: int,
) -> None:
    """
    Validate a batch of numpy data against the multimodal configuration using helpers.

    Args:
        numpy_batch: The batch of numpy data.
        multimodal_config: The OmegaConf DictConfig for the multimodal model.
        input_data_config: The OmegaConf DictConfig for the input data sources.
        expected_batch_size: The expected batch dimension size.

    Validates shapes and types. With the expected_batch_size provided, validates
    batch dimension against it. Does not infer or enforce cross-modality batch size
    consistency if expected_batch_size is None.
    """
    mm_config_dict: dict = OmegaConf.to_container(
        multimodal_config, resolve=True, throw_on_missing=True
    )
    input_data_dict: dict = OmegaConf.to_container(
        input_data_config, resolve=True, throw_on_missing=True
    )

    if not isinstance(expected_batch_size, int) or expected_batch_size <= 0:
        raise ValueError(
            f"expected_batch_size must be a positive integer, got {expected_batch_size}"
        )
    try:
        validate_static_config(mm_config_dict)
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Static configuration validation failed: {e}")
        raise

    log_msg = "Validating batch data shapes against configuration."
    log_msg += f" Expecting batch size: {expected_batch_size}."
    logger.info(log_msg)

    cfg = mm_config_dict

    if cfg.get("sat_encoder"):
        _validate_satellite_data(numpy_batch, cfg, input_data_dict, expected_batch_size)

    if cfg.get("nwp_encoders_dict"):
        _validate_nwp_data(numpy_batch, cfg, input_data_dict, expected_batch_size)

    if cfg.get("pv_encoder"):
        _validate_pv_data(numpy_batch, cfg, input_data_dict, expected_batch_size)

    if cfg.get("include_gsp_yield_history"):
        _validate_gsp_data(numpy_batch, cfg, input_data_dict, expected_batch_size)

    if cfg.get("include_sun"):
        _validate_sun_data(numpy_batch, cfg, input_data_dict, expected_batch_size)

    logger.info("Batch data shape validation checks completed.")
