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

    config_key_sat = "sat_encoder"
    if cfg.get(config_key_sat):
        key = "satellite_actual"
        logger.debug(f"Validating modality: {key}")
        data = check_batch_data(numpy_batch, key, np.ndarray, config_key_sat)
        time_res_mins = get_modality_interval(
            input_data_dict,
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
        nwp_interval_mins_config_dict = cfg.get("nwp_interval_minutes", {})

        for source in config_sources:
            source_key_str = f"{key}[{source}]"
            logger.debug(f"Validating NWP source: {source}")
            source_data_array = validate_nwp_source_structure(nwp_batch_data, source)
            
            time_res_mins = nwp_interval_mins_config_dict.get(source)
            if time_res_mins is None:
                logger.warning(
                    f"Time resolution for NWP source '{source}' not found in "
                    f"cfg['nwp_interval_minutes']. Defaulting to 60 minutes for validation."
                )
                time_res_mins = 60
            
            source_model_cfg = get_encoder_config(
                cfg, config_key_nwp, source_key_str, source_key=source
            )
            h = w = source_model_cfg["image_size_pixels"]
            c = source_model_cfg["in_channels"]
            hist_mins_source = nwp_hist_mins_dict.get(source)
            forecast_mins_source = nwp_forecast_mins_dict.get(source)

            if hist_mins_source is None:
                error_message = (
                    f"History minutes missing for NWP source '{source}' "
                    f"in cfg['nwp_history_minutes']."
                )
                raise KeyError(error_message)
            if forecast_mins_source is None:
                error_message = (
                    f"Forecast minutes missing for NWP source '{source}' "
                    f"in cfg['nwp_forecast_minutes']."
                )
                raise KeyError(error_message)

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
            (k_pv for k_pv in possible_keys if k_pv in input_data_dict and
                isinstance(input_data_dict.get(k_pv), dict)),
            None
        )
        if modality_data_cfg_key is None:
             raise KeyError(
                 "Neither 'site' nor 'pv' section found as a dictionary in "
                 "input_data_dict for PV validation"
             )
        logger.debug(f"Validating modality: {key} using input config '{modality_data_cfg_key}'")
        data = check_batch_data(numpy_batch, key, np.ndarray, config_key_pv)
        time_res_mins = get_modality_interval(input_data_dict, modality_data_cfg_key, None)
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
        time_res_mins = get_modality_interval(input_data_dict, "gsp", None)
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
        sun_time_res_mins = None
        try:
            sun_time_res_mins = get_modality_interval(input_data_dict, "sun", None)
            logger.debug(
                f"Using direct 'sun' configuration for time resolution: "
                f"{sun_time_res_mins} mins."
            )
        except (KeyError, ValueError) as e_sun_direct:
            logger.debug(
                f"Direct 'sun' config for interval not found or invalid. "
                f"Attempting fallback: {e_sun_direct}"
            )
            possible_fallback_keys = ["gsp", "site", "pv"]
            fallback_key_sun = next(
                (k_sun for k_sun in possible_fallback_keys if k_sun in input_data_dict and
                    isinstance(input_data_dict.get(k_sun), dict)),
                None
            )
            if fallback_key_sun is not None:
                try:
                    sun_time_res_mins = get_modality_interval(
                        input_data_dict, "sun", fallback_key_sun
                    )
                    logger.debug(
                        f"Using fallback '{fallback_key_sun}' for sun time resolution: "
                        f"{sun_time_res_mins} mins."
                    )
                except (KeyError, ValueError) as e_fallback:
                    raise KeyError(
                        "Cannot determine time resolution for sun. Direct 'sun' config failed, and "
                        f"fallback modality '{fallback_key_sun}' also failed or "
                        f"is invalid: {e_fallback}"
                    ) from e_fallback
            else:
                raise KeyError(
                     "Cannot determine time resolution for sun: No direct 'sun' config with "
                     "'time_resolution_minutes' found, and no suitable fallback modality ('gsp', "
                     "'site', or 'pv') found as a dictionary in input_data_dict."
                ) from e_sun_direct
        
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
