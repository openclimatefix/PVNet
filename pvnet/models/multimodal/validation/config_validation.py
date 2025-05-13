"""Main validation entry point for Multimodal configuration and batches."""

import logging

import numpy as np
from ocf_data_sampler.torch_datasets.sample.base import NumpyBatch

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


def validate(
    numpy_batch: NumpyBatch,
    multimodal_config: dict,
    input_data_config: dict,
    expected_batch_size: int,
) -> None:
    """
    Validate a batch of numpy data against the multimodal configuration using helpers.

    Validates shapes and types. With the expected_batch_size provided, validates
    batch dimension against it. Does not infer or enforce cross-modality batch size
    consistency if expected_batch_size is None.
    """
    if not isinstance(expected_batch_size, int) or expected_batch_size <= 0:
        raise ValueError(
            f"expected_batch_size must be a positive integer, got {expected_batch_size}"
        )
    try:
        validate_static_config(multimodal_config)
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Static configuration validation failed: {e}")
        raise

    log_msg = "Validating batch data shapes against configuration."
    log_msg += f" Expecting batch size: {expected_batch_size}."
    logger.info(log_msg)

    cfg = multimodal_config

    config_key_sat = "sat_encoder"
    if cfg.get(config_key_sat):
        key = "satellite_actual"
        logger.debug(f"Validating modality: {key}")
        data = check_batch_data(numpy_batch, key, np.ndarray, config_key_sat)
        time_res_mins = get_modality_interval(
            input_data_config,
            "satellite",
            secondary_modality_key=None
        )
        sat_cfg = get_encoder_config(cfg, config_key_sat, config_key_sat)
        h = w = sat_cfg["image_size_pixels"]
        c = sat_cfg["in_channels"]
        hist_mins = cfg["sat_history_minutes"]
        hist_steps, _ = get_time_steps(hist_mins, 0, time_res_mins)
        expected_shape_no_batch = (hist_steps, c, h, w)
        full_expected_shape = (expected_batch_size,) + expected_shape_no_batch
        validate_array_shape(
            data, full_expected_shape, key, time_res_mins
        )
        logger.debug(f"'{key}' shape validation passed (time resolution: {time_res_mins} mins).")

    config_key_nwp = "nwp_encoders_dict"
    if cfg.get(config_key_nwp):
        key = "nwp"
        logger.debug(f"Validating modality: {key}")
        nwp_batch_data = check_batch_data(numpy_batch, key, dict, config_key_nwp)
        nwp_model_cfg_dict = cfg[config_key_nwp]
        config_sources = set(nwp_model_cfg_dict.keys())
        batch_sources = set(nwp_batch_data.keys())
        if not config_sources.issubset(batch_sources):
            raise KeyError(
                f"NWP batch missing configured sources: {config_sources - batch_sources}"
            )
        if batch_sources - config_sources:
            logger.warning(
                f"NWP batch has extra sources not in config: {batch_sources - config_sources}"
            )
        nwp_hist_mins_dict = cfg["nwp_history_minutes"]
        nwp_forecast_mins_dict = cfg["nwp_forecast_minutes"]
        for source in config_sources:
            source_key_str = f"{key}[{source}]"
            logger.debug(f"Validating NWP source: {source}")
            source_data_array = validate_nwp_source_structure(nwp_batch_data, source)
            time_res_mins = get_modality_interval(input_data_config, "nwp", source)
            source_model_cfg = get_encoder_config(
                cfg, config_key_nwp, source_key_str, source_key=source
            )
            h = w = source_model_cfg["image_size_pixels"]
            c = source_model_cfg["in_channels"]
            hist_mins_source = nwp_hist_mins_dict.get(source)
            forecast_mins_source = nwp_forecast_mins_dict.get(source)
            hist_steps, forecast_steps = get_time_steps(
                hist_mins_source,
                forecast_mins_source,
                time_res_mins
            )
            expected_shape_no_batch = (hist_steps + forecast_steps, c, h, w)
            full_expected_shape = (expected_batch_size,) + expected_shape_no_batch
            validate_array_shape(
                source_data_array, full_expected_shape, source_key_str, time_res_mins
            )
            logger.debug(
                f"NWP source '{source}' shape validation passed "
                f"(time resolution: {time_res_mins} mins)."
            )

    config_key_pv = "pv_encoder"
    if cfg.get(config_key_pv):
        key = "pv"
        possible_keys = ["site", "pv"]
        modality_data_cfg_key = next(
            (k_pv for k_pv in possible_keys if k_pv in input_data_config and
                isinstance(input_data_config.get(k_pv), dict)),
            None
        )
        if modality_data_cfg_key is None:
             raise KeyError(
                 "Neither 'site' nor 'pv' section found as a dictionary in "
                 "input_data_config for PV validation"
             )
        logger.debug(f"Validating modality: {key} using input config '{modality_data_cfg_key}'")
        data = check_batch_data(numpy_batch, key, np.ndarray, config_key_pv)
        time_res_mins = get_modality_interval(input_data_config, modality_data_cfg_key, None)
        pv_cfg = get_encoder_config(cfg, config_key_pv, config_key_pv)
        num_sites = pv_cfg["num_sites"]
        hist_mins = cfg["pv_history_minutes"]
        hist_steps, _ = get_time_steps(hist_mins, 0, time_res_mins)
        expected_shape_no_batch = (hist_steps, num_sites)
        full_expected_shape = (expected_batch_size,) + expected_shape_no_batch
        validate_array_shape(
                data, full_expected_shape, key, time_res_mins,
                allow_ndim_plus_one=True
            )
        logger.debug(f"'{key}' shape validation passed (time resolution: {time_res_mins} mins).")

    config_key_gsp = "include_gsp_yield_history"
    if cfg.get(config_key_gsp):
        key = "gsp"
        logger.debug(f"Validating modality: {key}")
        data = check_batch_data(numpy_batch, key, np.ndarray, config_key_gsp)
        time_res_mins = get_modality_interval(input_data_config, "gsp", None)
        hist_steps, forecast_steps = get_time_steps(
            cfg["history_minutes"], cfg["forecast_minutes"], time_res_mins
        )
        expected_shape_no_batch = (hist_steps + forecast_steps,)
        full_expected_shape = (expected_batch_size,) + expected_shape_no_batch
        validate_array_shape(
            data,
            full_expected_shape,
            key,
            time_res_mins,
            allow_ndim_plus_one=True,
        )
        logger.debug(f"'{key}' shape validation passed (time resolution: {time_res_mins} mins).")

    config_key_sun = "include_sun"
    if cfg.get(config_key_sun):
        logger.debug("Validating modality: sun (azimuth, elevation)")
        possible_fallback_keys = ["gsp", "site", "pv"]
        fallback_key_sun = next(
            (k_sun for k_sun in possible_fallback_keys if k_sun in input_data_config and
                isinstance(input_data_config.get(k_sun), dict)),
            None
        )
        if fallback_key_sun is None:
            raise KeyError(
                 "Cannot determine interval for sun: No suitable fallback modality "
                 "('gsp', 'site', or 'pv') found as dictionary in input_data_config."
             )
        sun_time_res_mins = get_modality_interval(input_data_config, "sun", fallback_key_sun)
        hist_steps, forecast_steps = get_time_steps(
            cfg["history_minutes"], cfg["forecast_minutes"], sun_time_res_mins
        )
        expected_shape_no_batch = (hist_steps + forecast_steps,)
        full_expected_shape = (expected_batch_size,) + expected_shape_no_batch
        for key_sun_data in ["solar_azimuth", "solar_elevation"]:
            data_sun = check_batch_data(numpy_batch, key_sun_data, np.ndarray, config_key_sun)
            validate_array_shape(
                data_sun, full_expected_shape, key_sun_data, sun_time_res_mins
            )
        logger.debug(f"'sun' shape validation passed (time resolution: {sun_time_res_mins} mins).")

    logger.info("Batch data shape validation checks completed.")
