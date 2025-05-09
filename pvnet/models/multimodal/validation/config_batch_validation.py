"""Validation functoions for Multimodal batches."""

import logging
from typing import Any, Type

import numpy as np
from ocf_data_sampler.torch_datasets.sample.base import NumpyBatch

logger = logging.getLogger(__name__)


def check_batch_data(
    numpy_batch: NumpyBatch,
    batch_key: str,
    expected_type: Type,
    context_config_key: str,
) -> Any:
    """Check if modality data exists in batch and has the expected type.

    Args:
        numpy_batch: The batch of data (a dictionary-like object) to check.
        batch_key: The specific key within the batch whose data needs validation.
        expected_type: The Python type the data associated with `batch_key`
            is expected to be.
        context_config_key: The configuration key that requires this `batch_key`,
            used for providing context in error messages.

    Returns:
        The validated data corresponding to `batch_key` from `numpy_batch`.

    Raises:
        KeyError: If `batch_key` is not found in `numpy_batch`.
        TypeError: If the data for `batch_key` is not of `expected_type`.
    """
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


def get_modality_interval(
    input_data_config: dict,
    primary_modality_key: str,
    secondary_modality_key: str | None,
    is_nwp_source: bool = False,
) -> int:
    """Get and validate time resolution interval from input_data_config.

    Args:
        input_data_config: Dictionary containing configuration for data modalities,
            expected to hold 'time_resolution_minutes'.
        primary_modality_key: The main key for the modality (e.g., "satellite",
            "nwp", "sun") in `input_data_config`.
        secondary_modality_key: An optional secondary key. If `is_nwp_source`
            is True, this specifies the NWP source (e.g., "ecmwf"). If
            `primary_modality_key` is "sun", this can specify a fallback
            modality (e.g., "gsp") for interval lookup.
        is_nwp_source: Boolean flag. If True, indicates that the primary key
            is "nwp" and the `secondary_modality_key` refers to a specific
            NWP data source within the "nwp" section of the config.

    Returns:
        The validated time resolution in minutes for the specified modality.

    Raises:
        KeyError: If configuration keys are missing or not found.
        TypeError: If configuration values are not of the expected type (e.g., dict).
        ValueError: If `time_resolution_minutes` is invalid (e.g., not a
            positive integer), or if `secondary_modality_key` is required
            for NWP but not provided.
    """
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


def validate_array_shape(
    data: np.ndarray,
    expected_shape_with_batch: tuple,
    data_key: str,
    interval: int,
    allow_ndim_plus_one: bool = False,
) -> None:
    """Get and validate array dimensions.

    Args:
        data: The NumPy array whose shape is to be validated.
        expected_shape_with_batch: A tuple representing the expected shape
            of the array, including the batch dimension as the first element.
        data_key: The key or name identifying the data array, used for context
            in error messages.
        interval: The time interval associated with the data's time dimension,
            used for context in error messages.
        allow_ndim_plus_one: If True, allows the array's number of dimensions
            to be one greater than defined by `expected_shape_with_batch`
            (typically for an added channel dimension of size 1). Defaults to False.

    Raises:
        TypeError: If `data` is not a NumPy array.
        ValueError: If `data` is scalar, has a non-positive batch dimension,
            has an incorrect number of dimensions, or its shape does not match
            the expected shape.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"'{data_key}' data must be NumPy array, found {type(data).__name__}")
    if data.ndim == 0:
        raise ValueError(f"'{data_key}' array is scalar.")

    actual_batch_size = data.shape[0]
    if actual_batch_size <= 0:
        raise ValueError(f"'{data_key}' batch dimension has size <= 0: {actual_batch_size}")

    expected_batch_dim = expected_shape_with_batch[0]
    if actual_batch_size != expected_batch_dim:
        raise ValueError(
            f"Batch size mismatch for {data_key}: expected {expected_batch_dim}, "
            f"got {actual_batch_size}"
        )

    expected_ndim_base = len(expected_shape_with_batch)
    allowed_ndims = {expected_ndim_base}
    if allow_ndim_plus_one:
        allowed_ndims.add(expected_ndim_base + 1)

    actual_ndim = data.ndim

    if actual_ndim not in allowed_ndims:
        allowed_ndims_str = " or ".join(map(str, sorted(list(allowed_ndims))))
        raise ValueError(
            f"'{data_key}' dimension error. Expected {allowed_ndims_str} dims, Got {actual_ndim}"
        )

    if actual_ndim == expected_ndim_base:
        if data.shape != expected_shape_with_batch:
             raise ValueError(
                 f"'{data_key}' shape error using interval {interval}. "
                 f"Expected {expected_shape_with_batch}, Got {data.shape}"
             )
    elif actual_ndim == expected_ndim_base + 1:
        expected_shape_plus_one = expected_shape_with_batch + (1,)
        if data.shape != expected_shape_plus_one:
             raise ValueError(
                f"'{data_key}' shape error using interval {interval}. "
                f"Expected shape with extra dim {expected_shape_plus_one}, Got {data.shape}"
             )
    return


def validate_nwp_source_structure(
    nwp_batch_data: dict,
    source: str
) -> np.ndarray:
    """Validate structure of NWP source dict in batch and return data array.

    Args:
        nwp_batch_data: A dictionary where keys are NWP source identifiers (e.g.,
            "ecmwf") and values are dictionaries containing the NWP data array.
        source: The specific NWP source key (e.g., "ecmwf", "gfs") whose
            structure within `nwp_batch_data` needs validation.

    Returns:
        The validated NumPy array containing the NWP data for the specified `source`.

    Raises:
        KeyError: If the `source` is not in `nwp_batch_data` or if the
            expected "nwp" data array key is missing within the source's dict.
        TypeError: If the data for the `source` or the NWP array itself is not
            of the expected type (dict and np.ndarray, respectively).
    """
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


def get_time_steps(hist_mins: int, forecast_mins: int, interval: int) -> tuple:
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
