"""Main validation entry point for Multimodal configuration and batches."""

import logging
from typing import Any, Optional, Type

import numpy as np
from ocf_data_sampler.torch_datasets.sample.base import NumpyBatch

from pvnet.models.multimodal.validation.config_static_validation import (
    _validate_static_config,
    _get_encoder_config,
)

from pvnet.models.multimodal.validation.config_batch_validation import (
    _check_batch_data,
    _get_modality_interval,
    _validate_array_shape,
    _validate_nwp_source_structure,
    _get_time_steps,
)

logger = logging.getLogger(__name__)


def validate(
    numpy_batch: NumpyBatch,
    multimodal_config: dict,
    input_data_config: dict,
    expected_batch_size: Optional[int] = None,
) -> None:
    """Validate a batch of numpy data against the multimodal configuration using helpers."""
    try:
        _validate_static_config(multimodal_config)
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Static configuration validation failed: {e}")
        raise

    logger.info(
        f"Validating batch data shapes against configuration. Expected batch size: "
        f"{expected_batch_size if expected_batch_size is not None else 'Not Provided'}."
    )
    cfg = multimodal_config
    inferred_batch_size: int | None = None

    # Satellite
    config_key_sat = "sat_encoder"
    if cfg.get(config_key_sat):
        key = "satellite_actual"
        logger.debug(f"Validating modality: {key}")
        data = _check_batch_data(numpy_batch, key, np.ndarray, config_key_sat)
        interval = _get_modality_interval(input_data_config, "satellite")
        sat_cfg = _get_encoder_config(cfg, config_key_sat, config_key_sat)
        h = w = sat_cfg["image_size_pixels"]
        c = sat_cfg["in_channels"]
        hist_mins = cfg.get("sat_history_minutes", cfg["history_minutes"])
        hist_steps, _ = _get_time_steps(hist_mins, 0, interval)
        expected_shape_no_batch = (hist_steps, c, h, w)
        inferred_batch_size = _validate_array_shape(
            data, 5, expected_shape_no_batch, key, interval, inferred_batch_size
        )
        logger.debug(f"'{key}' validation passed with interval {interval}.")

    # NWP
    config_key_nwp = "nwp_encoders_dict"
    if cfg.get(config_key_nwp):
        key = "nwp"
        logger.debug(f"Validating modality: {key}")
        nwp_batch_data = _check_batch_data(numpy_batch, key, dict, config_key_nwp)
        nwp_model_cfg_dict = cfg[config_key_nwp]
        config_sources = set(nwp_model_cfg_dict.keys())
        batch_sources = set(nwp_batch_data.keys())
        if not config_sources.issubset(batch_sources):
            raise KeyError(f"NWP batch missing configured sources: {config_sources - batch_sources}")
        if batch_sources - config_sources:
            logger.warning(f"NWP batch has extra sources not in config: {batch_sources - config_sources}")
        nwp_hist_mins = cfg["nwp_history_minutes"]
        nwp_forecast_mins = cfg["nwp_forecast_minutes"]
        for source in config_sources:
            source_key_str = f"{key}[{source}]"
            logger.debug(f"Validating NWP source: {source}")
            source_data_array = _validate_nwp_source_structure(nwp_batch_data, source)
            interval = _get_modality_interval(input_data_config, "nwp", source, is_nwp_source=True)
            source_model_cfg = _get_encoder_config(cfg, config_key_nwp, source_key_str, source_key=source)
            h = w = source_model_cfg["image_size_pixels"]
            c = source_model_cfg["in_channels"]
            hist_min = nwp_hist_mins.get(source)
            forecast_min = nwp_forecast_mins.get(source)
            hist_steps, forecast_steps = _get_time_steps(hist_min, forecast_min, interval)
            expected_shape_no_batch = (hist_steps + forecast_steps, c, h, w)
            inferred_batch_size = _validate_array_shape(
                source_data_array, 5, expected_shape_no_batch, source_key_str, interval, inferred_batch_size
            )
            logger.debug(f"NWP source '{source}' validation passed with interval {interval}.")

    # PV
    config_key_pv = "pv_encoder"
    if cfg.get(config_key_pv):
        key = "pv"
        modality_data_cfg_key = next((k for k in ["site", "pv"] if k in input_data_config), None)
        if modality_data_cfg_key is None:
             raise KeyError("Neither 'site' nor 'pv' section found in input_data_config for PV validation")
        logger.debug(f"Validating modality: {key} using input config '{modality_data_cfg_key}'")
        data = _check_batch_data(numpy_batch, key, np.ndarray, config_key_pv)
        interval = _get_modality_interval(input_data_config, modality_data_cfg_key)
        pv_cfg = _get_encoder_config(cfg, config_key_pv, config_key_pv)
        num_sites = pv_cfg["num_sites"]
        hist_mins = cfg.get("pv_history_minutes", cfg["history_minutes"])
        hist_steps, _ = _get_time_steps(hist_mins, 0, interval)
        expected_shape_no_batch = (hist_steps, num_sites)
        inferred_batch_size = _validate_array_shape(
            data, 3, expected_shape_no_batch, key, interval, inferred_batch_size
        )
        logger.debug(f"'{key}' validation passed with interval {interval}.")

    # GSP
    config_key_gsp = "include_gsp_yield_history"
    if cfg.get(config_key_gsp):
        key = "gsp"
        logger.debug(f"Validating modality: {key}")
        data = _check_batch_data(numpy_batch, key, np.ndarray, config_key_gsp)
        interval = _get_modality_interval(input_data_config, "gsp")
        hist_steps, forecast_steps = _get_time_steps(
            cfg["history_minutes"], cfg["forecast_minutes"], interval
        )
        expected_shape_no_batch = (hist_steps + forecast_steps,)
        inferred_batch_size = _validate_array_shape(
            data, 2, expected_shape_no_batch, key, interval, inferred_batch_size, allow_ndim_plus_one=True
        )
        logger.debug(f"'{key}' validation passed with interval {interval}.")

    # Solar
    config_key_sun = "include_sun"
    if cfg.get(config_key_sun):
        logger.debug("Validating modality: sun (azimuth, elevation)")
        fallback_key = next((k for k in ["gsp", "site"] if k in input_data_config), None)
        sun_interval = _get_modality_interval(input_data_config, "sun", fallback_key)
        hist_steps, forecast_steps = _get_time_steps(
            cfg["history_minutes"], cfg["forecast_minutes"], sun_interval
        )
        expected_shape_no_batch = (hist_steps + forecast_steps,)
        for key_sun in ["solar_azimuth", "solar_elevation"]:
            data_sun = _check_batch_data(numpy_batch, key_sun, np.ndarray, config_key_sun)
            inferred_batch_size = _validate_array_shape(
                data_sun, 2, expected_shape_no_batch, key_sun, sun_interval, inferred_batch_size
            )
        logger.debug(f"'sun' validation passed with interval {sun_interval}.")

    # Final Batch Size Check
    if inferred_batch_size is None and expected_batch_size is None:
        logger.warning("Batch size could not be determined (no modalities checked or empty batch?).")
    elif (
        inferred_batch_size is not None
        and expected_batch_size is not None
        and inferred_batch_size != expected_batch_size
    ):
         raise ValueError(
             f"Batch size inconsistency detected. Inferred batch size "
             f"{inferred_batch_size} from modalities does not match expected_batch_size "
             f"{expected_batch_size}."
         )
    elif inferred_batch_size is not None:
         logger.info(f"All modalities passed validation with consistent inferred batch size: {inferred_batch_size}")
    elif expected_batch_size is not None:
         logger.warning(f"Expected batch size {expected_batch_size} provided, but no data modalities were validated to confirm.")

    logger.info("Batch data shape validation successful against configuration.")
