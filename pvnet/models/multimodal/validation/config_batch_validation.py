"""Validation functoions for Multimodal batches."""

import logging
from typing import Any, Optional, Type

import numpy as np
from ocf_data_sampler.torch_datasets.sample.base import NumpyBatch

logger = logging.getLogger(__name__)


def _check_batch_data(
    numpy_batch: NumpyBatch,
    batch_key: str,
    expected_type: Type,
    context_config_key: str,
) -> Any:
    """Check if modality data exists in batch and has the expected type."""
    if batch_key not in numpy_batch:
        raise KeyError(
            f"Batch missing required '{batch_key}' data "
            f"(required by config key '{context_config_key}')."
        )
    data: Any = numpy_batch[batch_key]
    if not isinstance(data, expected_type):
        expected_type_name = getattr(expected_type, '__name__', str(expected_type))
        raise TypeError(
            f"'{batch_key}' data must be {expected_type_name}, "
            f"found {type(data).__name__}"
        )
    return data


def _get_modality_interval(
    input_data_config: dict,
    primary_modality_key: str,
    secondary_modality_key: Optional[str] = None,
    is_nwp_source: bool = False,
) -> int:
    """Get and validate time resolution interval from input_data_config."""
    config_to_check = input_data_config
    lookup_key = primary_modality_key
    error_context = f"modality '{primary_modality_key}'"

    if is_nwp_source:
        if primary_modality_key not in config_to_check:
            raise KeyError(f"NWP section ('{primary_modality_key}') not found in input_data_config")
        config_to_check = config_to_check[primary_modality_key]
        if not isinstance(config_to_check, dict):
            raise TypeError(f"Expected dict for '{primary_modality_key}' in input_data_config")
        if secondary_modality_key is None:
            raise ValueError(
                "secondary_modality_key (NWP source) is required when is_nwp_source=True"
            )

        lookup_key = secondary_modality_key
        error_context = f"NWP source '{lookup_key}'"

    modality_config_dict = None
    if lookup_key not in config_to_check:
        if (primary_modality_key == "sun"
                and secondary_modality_key is not None
                and secondary_modality_key in input_data_config):
            logger.debug(f"Using '{secondary_modality_key}' interval as fallback for sun.")
            lookup_key = secondary_modality_key
            error_context = f"fallback modality '{lookup_key}' for sun"
            modality_config_dict = input_data_config.get(lookup_key)
        else:
             raise KeyError(f"Could not find config for {error_context} in input_data_config")
    else:
         modality_config_dict = config_to_check.get(lookup_key)


    if not isinstance(modality_config_dict, dict):
        raise TypeError(
            f"Expected dict for {error_context} config, "
            f"got {type(modality_config_dict).__name__}"
        )
    try:
        interval = modality_config_dict['time_resolution_minutes']
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("Interval must be a positive integer")
        return interval
    except KeyError:
        raise KeyError(
            f"Could not find 'time_resolution_minutes' for {error_context} "
            f"in input_data_config"
        )
    except ValueError as e:
        raise ValueError(f"Invalid time_resolution_minutes for {error_context}: {e}")


def _validate_array_shape(
    data: np.ndarray,
    expected_ndim: int,
    expected_shape_no_batch: tuple,
    data_key: str,
    interval: int,
    current_inferred_batch_size: Optional[int],
    allow_ndim_plus_one: bool = False,
) -> int:
    """
    Validate ndim, full shape (incl. batch dim), and batch consistency.

    Returns the validated batch size inferred from the data.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"'{data_key}' data must be NumPy array, found {type(data).__name__}")
    if data.ndim == 0:
        raise ValueError(f"'{data_key}' array is scalar.")

    actual_batch_size = data.shape[0]
    if actual_batch_size <= 0:
         raise ValueError(f"'{data_key}' batch dimension has size <= 0: {actual_batch_size}")

    if current_inferred_batch_size is None:
        batch_size_to_use = actual_batch_size
        logger.debug(f"Inferred batch size: {batch_size_to_use} from {data_key}")
    elif current_inferred_batch_size != actual_batch_size:
        raise ValueError(
            f"Batch size mismatch for {data_key}: expected {current_inferred_batch_size}, "
            f"got {actual_batch_size}"
        )
    else:
        batch_size_to_use = current_inferred_batch_size

    expected_shape_with_batch = (batch_size_to_use,) + expected_shape_no_batch
    actual_ndim = data.ndim
    shape_matches = data.shape == expected_shape_with_batch

    if allow_ndim_plus_one and actual_ndim == expected_ndim + 1:
        expected_shape_plus_one = expected_shape_with_batch + (1,)
        if data.shape == expected_shape_plus_one:
            return batch_size_to_use
        else:
             raise ValueError(
                f"'{data_key}' shape error using interval {interval}. "
                f"Expected {expected_shape_with_batch} or {expected_shape_plus_one}, "
                f"Got {data.shape}"
            )

    elif shape_matches:
        return batch_size_to_use

    elif actual_ndim != expected_ndim:
         allowed_ndims_str = (
             f"{expected_ndim} or {expected_ndim + 1}"
             if allow_ndim_plus_one else str(expected_ndim)
         )
         raise ValueError(
             f"'{data_key}' dimension error. Expected {allowed_ndims_str} dims, Got {actual_ndim}"
         )
    else:
         raise ValueError(
             f"'{data_key}' shape error using interval {interval}. "
             f"Expected {expected_shape_with_batch}, Got {data.shape}"
         )


def _validate_nwp_source_structure(
    nwp_batch_data: dict,
    source: str
) -> np.ndarray:
    """Validate structure of NWP source dict in batch and return data array."""
    if source not in nwp_batch_data:
        raise KeyError(f"NWP data for configured source '{source}' is missing in batch dict.")

    source_data_dict = nwp_batch_data[source]
    if not isinstance(source_data_dict, dict):
        raise TypeError(
            f"NWP data for source '{source}' must be a dict, "
            f"found {type(source_data_dict).__name__}"
        )

    data_array_key = "nwp"
    if data_array_key not in source_data_dict:
        raise KeyError(
            f"NWP data array key '{data_array_key}' not found in source dict for '{source}'"
        )

    source_data_array = source_data_dict[data_array_key]
    if not isinstance(source_data_array, np.ndarray):
        raise TypeError(
            f"NWP data array for source '{source}' (key '{data_array_key}') must be np.ndarray, "
            f"found {type(source_data_array).__name__}"
        )
    return source_data_array


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
