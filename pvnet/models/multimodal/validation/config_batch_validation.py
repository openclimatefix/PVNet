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
) -> int:
    """Get and validate time resolution interval from input_data_config.

    Args:
        input_data_config: Dictionary containing configuration for data modalities,
            expected to hold 'time_resolution_minutes'.
        primary_modality_key: The main key for the modality in `input_data_config`.
            If this is "nwp", then `secondary_modality_key` is treated as the
            specific NWP source name within the "nwp" section.
        secondary_modality_key: An optional secondary key.
            - If `primary_modality_key` is "nwp", this is REQUIRED and specifies
              the NWP source (e.g., "ecmwf").
            - If `primary_modality_key` is "sun", this can specify a fallback
              modality (e.g., "gsp") for interval lookup if the "sun" config
              is not found directly.
            - Otherwise, typically None for non-NWP, non-sun direct lookups.

    Returns:
        The validated time resolution in minutes for the specified modality.

    Raises:
        KeyError: If configuration keys are missing or not found.
        TypeError: If configuration values are not of the expected type (e.g., dict).
        ValueError: If `time_resolution_minutes` is invalid (e.g., not a
            positive integer), or if `secondary_modality_key` is required
            for NWP but not provided.
    """
    config_section_to_search = input_data_config
    key_for_config = primary_modality_key
    error_context_desc = f"modality '{primary_modality_key}'"

    if primary_modality_key == "nwp":
        if secondary_modality_key is None:
            raise ValueError(
                "When primary_modality_key is 'nwp', secondary_modality_key "
                "(the NWP source name) must be provided."
            )
        if "nwp" not in input_data_config or not isinstance(input_data_config["nwp"], dict):
                raise KeyError(
                    "NWP section ('nwp') not found or is not a dictionary "
                    "in input_data_config."
                )

        config_section_to_search = input_data_config["nwp"]
        key_for_config = secondary_modality_key
        error_context_desc = f"NWP source '{key_for_config}'"

    modality_config_dict = config_section_to_search.get(key_for_config)

    if modality_config_dict is None:
        if primary_modality_key == "sun" and secondary_modality_key is not None:
            logger.debug(
                f"Config for '{primary_modality_key}' not found directly, "
                f"attempting fallback to '{secondary_modality_key}' "
                f"from top-level input_data_config."
            )
            modality_config_dict = input_data_config.get(secondary_modality_key)
            if modality_config_dict is not None:
                error_context_desc = f"fallback modality '{secondary_modality_key}' for sun"
            else:
                raise KeyError(
                    f"Could not find config for modality '{primary_modality_key}' or "
                    f"fallback '{secondary_modality_key}' in input_data_config."
                )
        else:
            raise KeyError(
                f"Could not find config for {error_context_desc} "
                f"in the relevant section."
            )

    if not isinstance(modality_config_dict, dict):
        raise TypeError(
            f"Expected a dictionary for {error_context_desc} config, "
            f"got {type(modality_config_dict).__name__}."
        )
    try:
        resolution_minutes = modality_config_dict['time_resolution_minutes']
        if not isinstance(resolution_minutes, int) or resolution_minutes <= 0:
            raise ValueError(
                f"'time_resolution_minutes' for {error_context_desc} "
                "must be a positive integer, "
                f"found {resolution_minutes}."
            )
        return resolution_minutes
    except KeyError:
        raise KeyError(
            f"Could not find 'time_resolution_minutes' for {error_context_desc} "
            f"in its configuration."
        )
    except ValueError as e:
        raise ValueError(f"Invalid 'time_resolution_minutes' for {error_context_desc}: {e}")


def validate_array_shape(
    data: np.ndarray,
    expected_shape_with_batch: tuple,
    data_key: str,
    time_resolution_minutes: int,
    allow_ndim_plus_one: bool = False,
) -> None:
    """Validate array dimensions, checking batch size, ndim, and per-dimension sizes.

    Args:
        data: The NumPy array whose shape is to be validated.
        expected_shape_with_batch: A tuple representing the expected shape
            of the array, including the batch dimension as the first element.
        data_key: The key or name identifying the data array, used for context
            in error messages.
        time_resolution_minutes: The time resolution (in minutes) of the data,
            used for context in error messages.
        allow_ndim_plus_one: If True, allows the array's number of dimensions
            to be one greater than defined by `expected_shape_with_batch`
            (typically for an added channel dimension of size 1). Defaults to False.

    Raises:
        TypeError: If `data` is not a NumPy array.
        ValueError: If `data` is scalar, has a non-positive batch dimension,
            has an incorrect number of dimensions, or its shape does not match
            the expected shape at any dimension.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"'{data_key}' data must be NumPy array, found {type(data).__name__}.")
    if data.ndim == 0:
        raise ValueError(f"'{data_key}' array is scalar.")

    actual_batch_size = data.shape[0]
    if actual_batch_size <= 0:
        raise ValueError(f"'{data_key}' batch dimension has size <= 0: {actual_batch_size}.")

    expected_batch_dim_value = expected_shape_with_batch[0]
    if actual_batch_size != expected_batch_dim_value:
        raise ValueError(
            f"Batch size mismatch for '{data_key}'. Expected {expected_batch_dim_value}, "
            f"Got {actual_batch_size}. "
            f"(Time resolution for context: {time_resolution_minutes} mins)."
        )

    expected_ndim_base = len(expected_shape_with_batch)
    allowed_ndims = {expected_ndim_base}
    if allow_ndim_plus_one:
        allowed_ndims.add(expected_ndim_base + 1)

    actual_ndim = data.ndim
    if actual_ndim not in allowed_ndims:
        allowed_ndims_str = " or ".join(map(str, sorted(list(allowed_ndims))))
        raise ValueError(
            f"'{data_key}' dimension count error. "
            f"Expected {allowed_ndims_str} dims, Got {actual_ndim}. "
            f"(Time resolution for context: {time_resolution_minutes} mins)."
        )

    final_expected_shape: tuple
    if actual_ndim == expected_ndim_base:
        final_expected_shape = expected_shape_with_batch
    else:
        final_expected_shape = expected_shape_with_batch + (1,)

    for i in range(1, actual_ndim):
        if data.shape[i] != final_expected_shape[i]:
            raise ValueError(
                f"'{data_key}' shape error at dimension {i} (0-indexed, after batch). "
                f"Expected size {final_expected_shape[i]}, Got {data.shape[i]}. "
                f"Full Expected Shape: {final_expected_shape}, Full Actual Shape: {data.shape}. "
                f"(Time resolution for context: {time_resolution_minutes} mins)."
            )
    
    if data.shape != final_expected_shape:
         raise ValueError(
             f"'{data_key}' general shape mismatch (unexpected). "
             f"Expected {final_expected_shape}, Got {data.shape}. "
             f"(Time resolution for context: {time_resolution_minutes} mins)."
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
