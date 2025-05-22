"""Tests for multimodal configuration and batch validation functions."""

import logging
from copy import deepcopy

import numpy as np
import pytest
import re

from omegaconf import OmegaConf

from pvnet.models.multimodal.validation.config_validation import validate

NumpyBatch = dict
logger = logging.getLogger(__name__)


def test_validate_valid_inputs(
    valid_config_dict,
    sample_numpy_batch,
    valid_input_data_config
):
    """Test validate with valid config and correctly shaped batch."""
    try:
        omega_multimodal_config = OmegaConf.create(valid_config_dict)
        omega_input_data_config = OmegaConf.create(valid_input_data_config)

        validate(
            numpy_batch=sample_numpy_batch,
            multimodal_config=omega_multimodal_config,
            input_data_config=omega_input_data_config,
            expected_batch_size=4
        )
    except Exception as e:
        pytest.fail(f"validate raised an unexpected exception with valid inputs: {e}")


@pytest.mark.parametrize(
    "key_to_delete, expected_error_match",
    [
        ("forecast_minutes", r"missing required key: 'forecast_minutes'"),
        ("output_network", r"missing required section: 'output_network'"),
    ],
    ids=[
        "missing_top_level_key",
        "missing_required_section",
    ],
)
def test_validate_static_error_missing_required_item(
    valid_config_dict,
    key_to_delete,
    expected_error_match,
    valid_input_data_config
):
    """Tests validate catches static KeyError for missing required config items."""
    invalid_cfg_dict = deepcopy(valid_config_dict)
    dummy_batch = {}

    if key_to_delete in invalid_cfg_dict:
        del invalid_cfg_dict[key_to_delete]
    else:
        pytest.skip(f"Key '{key_to_delete}' not in valid_config_dict fixture.")

    omega_invalid_cfg = OmegaConf.create(invalid_cfg_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    with pytest.raises(KeyError, match=expected_error_match):
        validate(
            dummy_batch,
            omega_invalid_cfg,
            omega_input_data_config,
            expected_batch_size=1
        )


@pytest.mark.parametrize(
    "section_name, invalid_value",
    [
        ("optimizer", "not-a-dictionary"),
        ("sat_encoder", ["list", "not", "dict"]),
    ],
    ids=[
        "required_section_wrong_type",
        "optional_section_wrong_type",
    ],
)
def test_validate_static_error_section_wrong_type(
    valid_config_dict,
    section_name,
    invalid_value,
    valid_input_data_config
):
    """Tests validate catches static TypeError for sections not being dicts."""
    invalid_cfg_dict = deepcopy(valid_config_dict)
    dummy_batch = {}
    if section_name not in invalid_cfg_dict:
        pytest.skip(f"'{section_name}' not in fixture.")

    invalid_cfg_dict[section_name] = invalid_value
    match_str = rf"section '{section_name}'.*must be a dictionary"

    omega_invalid_cfg = OmegaConf.create(invalid_cfg_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    with pytest.raises(TypeError, match=match_str):
        validate(
            dummy_batch,
            omega_invalid_cfg,
            omega_input_data_config,
            expected_batch_size=1
        )


@pytest.mark.parametrize(
    "section_name, invalid_sub_dict",
    [
        ("output_network", {"some_other_key": 123}),
        ("sat_encoder", {"another_key": 456}),
    ],
    ids=[
        "required_section_missing_target",
        "optional_section_missing_target",
    ],
)
def test_validate_static_error_section_missing_target(
    valid_config_dict,
    section_name,
    invalid_sub_dict,
    valid_input_data_config
):
    """Tests validate catches static KeyError when section dict misses '_target_'."""
    invalid_cfg_dict = deepcopy(valid_config_dict)
    dummy_batch = {}
    if section_name not in invalid_cfg_dict:
        pytest.skip(f"'{section_name}' not in fixture.")

    invalid_cfg_dict[section_name] = invalid_sub_dict
    match_str = rf"section '{section_name}'.*missing required sub-key: '_target_'"

    omega_invalid_cfg = OmegaConf.create(invalid_cfg_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    with pytest.raises(KeyError, match=match_str):
        validate(
            dummy_batch,
            omega_invalid_cfg,
            omega_input_data_config,
            expected_batch_size=1
        )


def test_validate_static_error_nwp_sub_item_missing_target(
    valid_config_dict,
    valid_input_data_config
):
    """Tests validate catches static KeyError for NWP sub-item missing '_target_'."""
    invalid_cfg_dict = deepcopy(valid_config_dict)
    dummy_batch = {}
    if not invalid_cfg_dict.get("nwp_encoders_dict"):
        pytest.skip("nwp_encoders_dict missing or empty in fixture.")

    invalid_cfg_dict["nwp_encoders_dict"] = deepcopy(invalid_cfg_dict["nwp_encoders_dict"])
    try:
        nwp_key = list(invalid_cfg_dict["nwp_encoders_dict"].keys())[0]
        invalid_cfg_dict["nwp_encoders_dict"][nwp_key] = {"wrong_key": 789}
    except IndexError:
        pytest.skip("nwp_encoders_dict is empty in fixture.")

    match_str = rf"Config for NWP source '{nwp_key}' missing required key: '_target_'"

    omega_invalid_cfg = OmegaConf.create(invalid_cfg_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    with pytest.raises(KeyError, match=match_str):
        validate(
            dummy_batch,
            omega_invalid_cfg,
            omega_input_data_config,
            expected_batch_size=1
        )


def test_validate_static_error_missing_req_time_param(
    valid_config_dict,
    valid_input_data_config
):
    """Tests validate catches static KeyError for missing dependent time param."""
    invalid_cfg_dict = deepcopy(valid_config_dict)
    dummy_batch = {}
    if not invalid_cfg_dict.get("sat_encoder"):
        pytest.skip("sat_encoder not configured in fixture.")

    if "sat_history_minutes" in invalid_cfg_dict:
        del invalid_cfg_dict["sat_history_minutes"]
    else:
        pytest.skip("'sat_history_minutes' not in fixture.")

    match_str = r"includes 'sat_encoder' but is missing 'sat_history_minutes'"

    omega_invalid_cfg = OmegaConf.create(invalid_cfg_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    with pytest.raises(KeyError, match=match_str):
        validate(
            dummy_batch,
            omega_invalid_cfg,
            omega_input_data_config,
            expected_batch_size=1
        )


def test_validate_static_error_nwp_time_keys_mismatch(
    valid_config_dict,
    valid_input_data_config
):
    """Tests validate catches static ValueError for NWP time key mismatch."""
    invalid_cfg_dict = deepcopy(valid_config_dict)
    dummy_batch = {}
    if not invalid_cfg_dict.get("nwp_encoders_dict") or not invalid_cfg_dict.get(
        "nwp_history_minutes"
    ):
        pytest.skip("NWP sections missing or empty in fixture.")

    invalid_cfg_dict["nwp_history_minutes"] = deepcopy(invalid_cfg_dict["nwp_history_minutes"])
    if "dummy_source_for_mismatch" not in invalid_cfg_dict["nwp_history_minutes"]:
        invalid_cfg_dict["nwp_history_minutes"]["dummy_source_for_mismatch"] = 60
    else:
        pytest.skip("Cannot reliably create mismatch key.")

    match_str = r"Keys in 'nwp_history_minutes'.*do not match sources"

    omega_invalid_cfg = OmegaConf.create(invalid_cfg_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    with pytest.raises(ValueError, match=match_str):
        validate(
            dummy_batch,
            omega_invalid_cfg,
            omega_input_data_config,
            expected_batch_size=1
        )


def test_validate_batch_error_missing_modality_key(
    valid_config_dict,
    sample_numpy_batch,
    valid_input_data_config
):
    """Test KeyError when a required modality is missing from the batch."""
    config_dict = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)
    if not config_dict.get("sat_encoder"):
        pytest.skip("sat_encoder not in config fixture.")

    key_to_check = "satellite_actual"
    if key_to_check in batch:
        del batch[key_to_check]
    elif config_dict.get("sat_encoder"):
        pytest.skip(
            f"Config expects {key_to_check}, but sample_numpy_batch missing it."
        )

    match_str = f"Batch missing required '{key_to_check}' data"

    omega_config = OmegaConf.create(config_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    with pytest.raises(KeyError, match=match_str):
        validate(
            batch,
            omega_config,
            omega_input_data_config,
            expected_batch_size=4
        )


def test_validate_batch_error_modality_wrong_type(
    valid_config_dict,
    sample_numpy_batch,
    valid_input_data_config
):
    """Test TypeError when batch data for a modality is not a numpy array."""
    config_dict = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)
    key_to_check = "satellite_actual"

    if not config_dict.get("sat_encoder"):
        pytest.skip("sat_encoder not in config fixture.")
    if key_to_check not in batch and config_dict.get("sat_encoder"):
        pytest.skip(
            f"Config expects {key_to_check}, but sample_numpy_batch missing it."
        )
    elif key_to_check not in batch:
        pytest.skip(f"{key_to_check} not in batch, cannot test wrong type.")

    batch[key_to_check] = "this is not a numpy array"

    match_str = f"'{key_to_check}' data must be ndarray, found str"

    omega_config = OmegaConf.create(config_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    with pytest.raises(TypeError, match=match_str):
        validate(
            batch,
            omega_config,
            omega_input_data_config,
            expected_batch_size=4
        )


def test_validate_batch_error_wrong_ndim(
    valid_config_dict,
    sample_numpy_batch,
    valid_input_data_config
):
    """Test ValueError for incorrect number of dimensions (ndim) in batch data."""
    config_dict = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)
    key_to_check = "satellite_actual"

    if not config_dict.get("sat_encoder"):
        pytest.skip("sat_encoder not in config fixture.")
    if key_to_check not in batch and config_dict.get("sat_encoder"):
        pytest.skip(
            f"Config expects {key_to_check}, but sample_numpy_batch missing it."
        )
    elif key_to_check not in batch:
        pytest.skip(f"{key_to_check} not in batch, cannot test wrong ndim.")

    correct_shape = batch[key_to_check].shape
    actual_batch_size = correct_shape[0]

    if len(correct_shape) <= 1:
        pytest.skip("Cannot reduce ndim further.")

    try:
        wrong_ndim_shape = correct_shape[:-1]
        if len(wrong_ndim_shape) < 2:
             pytest.skip("Cannot create reliably incorrect ndim.")
        batch[key_to_check] = np.zeros(wrong_ndim_shape)
        logger.info(f"Modified '{key_to_check}' shape to {wrong_ndim_shape} for ndim test.")
    except IndexError:
        pytest.skip("Cannot modify array dimensions reliably for test.")

    match_str = rf"'{key_to_check}' dimension count error\. Expected {len(correct_shape)} dims, Got {len(wrong_ndim_shape)}\."

    omega_config = OmegaConf.create(config_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    with pytest.raises(ValueError, match=match_str):
        validate(
            batch,
            omega_config,
            omega_input_data_config,
            expected_batch_size=actual_batch_size
        )


def test_validate_batch_error_wrong_shape_time(
    valid_config_dict,
    sample_numpy_batch,
    valid_input_data_config
):
    """Test ValueError for incorrect time dimension size."""
    config_dict = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)
    key_to_check = "satellite_actual"

    if not config_dict.get("sat_encoder"):
        pytest.skip("sat_encoder not in config fixture.")
    if key_to_check not in batch and config_dict.get("sat_encoder"):
        pytest.skip(
            f"Config expects {key_to_check}, but sample_numpy_batch missing it."
        )
    elif key_to_check not in batch:
        pytest.skip(f"{key_to_check} not in batch, cannot test wrong time shape.")

    correct_shape = batch[key_to_check].shape
    if len(correct_shape) < 2:
        pytest.skip("Batch has insufficient dimensions.")
    wrong_time_shape = list(correct_shape)
    wrong_time_shape[1] += 1
    batch[key_to_check] = np.zeros(tuple(wrong_time_shape))

    omega_config = OmegaConf.create(config_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    match_str = (
        rf"'{key_to_check}' shape error for time_steps\. "
        rf"Expected size {correct_shape[1]}, Got {wrong_time_shape[1]}\."
    )
    with pytest.raises(ValueError, match=match_str):
        validate(
            batch,
            omega_config,
            omega_input_data_config,
            expected_batch_size=4
        )


def test_validate_batch_error_wrong_shape_spatial(
    valid_config_dict,
    sample_numpy_batch,
    valid_input_data_config
):
    """Test ValueError for incorrect spatial dimension size."""
    config_as_dict = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)
    key_to_check = "satellite_actual"

    if not config_as_dict.get("sat_encoder"):
        pytest.skip("sat_encoder not in config fixture.")
    if key_to_check not in batch and config_as_dict.get("sat_encoder"):
        pytest.skip(
            f"Config expects {key_to_check}, but sample_numpy_batch missing it."
        )
    elif key_to_check not in batch:
        pytest.skip(f"{key_to_check} not in batch, cannot test wrong spatial shape.")

    correct_shape = batch[key_to_check].shape
    if len(correct_shape) < 4:
        pytest.skip("Batch has insufficient dimensions for spatial check.")
    wrong_spatial_shape = list(correct_shape)
    wrong_spatial_shape[3] += 1
    batch[key_to_check] = np.zeros(tuple(wrong_spatial_shape))

    omega_config = OmegaConf.create(config_as_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    match_str = (
        rf"'{key_to_check}' shape error for height\. "
        rf"Expected size {correct_shape[3]}, Got {wrong_spatial_shape[3]}\."
    )
    with pytest.raises(ValueError, match=match_str):
        validate(
            batch,
            omega_config,
            omega_input_data_config,
            expected_batch_size=4
        )


def _get_batch_size_test_helper(batch_dict, key):
    """Helper (for testing) to get batch size, handling NWP dict structure."""
    if key not in batch_dict:
        raise KeyError(f"Key '{key}' not found in batch for batch size check.")
    data = batch_dict[key]

    if key == "nwp":
        if not isinstance(data, dict) or not data:
            raise TypeError(f"NWP data for key '{key}' in batch is not a non-empty dict")
        first_source_key = next(iter(data.keys()))
        first_source_dict = data[first_source_key]
        if not isinstance(first_source_dict, dict):
            raise TypeError(
                f"NWP source value for '{first_source_key}' is not a dict: {first_source_dict}"
            )
        try:
            data_array = next(
                v
                for v in first_source_dict.values()
                if isinstance(v, np.ndarray) and v.ndim > 0
            )
        except StopIteration:
            raise ValueError(
                f"No numpy array with ndim > 0 found within NWP source dict: {first_source_dict}"
            )
        return data_array.shape[0]
    elif isinstance(data, np.ndarray):
        if data.ndim == 0:
            raise ValueError(f"Data for key '{key}' is a 0-dim array, cannot get batch size.")
        return data.shape[0]
    else:
        raise TypeError(
            f"Data for key '{key}' ({type(data)}) is not a numpy array or NWP dict."
        )


def test_validate_error_mismatch_expected_batch_size(
    valid_config_dict,
    sample_numpy_batch,
    valid_input_data_config
):
    """Test ValueError when expected_batch_size mismatches the actual batch size."""
    config_as_dict = valid_config_dict
    batch = sample_numpy_batch
    try:
        actual_batch_size_from_fixture = _get_batch_size_test_helper(batch, "gsp")
    except Exception as e:
        pytest.skip(f"Could not determine actual batch size from fixture: {e}")
    if actual_batch_size_from_fixture <= 0:
         pytest.skip("Fixture batch size is not positive.")
    incorrect_expected_size = actual_batch_size_from_fixture + 1

    omega_config = OmegaConf.create(config_as_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    with pytest.raises(ValueError) as exc_info:
        validate(
            batch,
            omega_config,
            omega_input_data_config,
            expected_batch_size=incorrect_expected_size
        )
    error_message = str(exc_info.value)

    expected_data_key_in_error = "satellite_actual"
    actual_runtime_batch_size_for_error = actual_batch_size_from_fixture

    expected_message_pattern = (
        rf"Mismatch for '{expected_data_key_in_error}' in batch_size\. "
        rf"Expected size {incorrect_expected_size}, Got {actual_runtime_batch_size_for_error}\."
    )
    assert re.search(expected_message_pattern, error_message), \
        f"Pattern <{expected_message_pattern}> not found in error: <{error_message}>"


def test_validate_error_internal_mismatch_with_expected_size(
    valid_config_dict,
    sample_numpy_batch,
    valid_input_data_config
):
    """Test ValueError when a modality's batch size internally mismatches expected_batch_size."""
    config_dict = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)
    expected_data_keys = set()
    if config_dict.get("sat_encoder"):
        expected_data_keys.add("satellite_actual")
    if config_dict.get("nwp_encoders_dict"):
        expected_data_keys.add("nwp")
    if config_dict.get("pv_encoder"):
        expected_data_keys.add("pv")
    if config_dict.get("include_gsp_yield_history"):
        expected_data_keys.add("gsp")
    if config_dict.get("include_sun"):
        expected_data_keys.add("solar_azimuth")
        expected_data_keys.add("solar_elevation")

    modality_batch_sizes = {}
    for k in expected_data_keys:
        if k not in batch:
            logger.info(f"Expected key '{k}' not in batch for size check.")
            continue
        try:
            modality_batch_sizes[k] = _get_batch_size_test_helper(batch, k)
        except (TypeError, ValueError, KeyError, StopIteration) as e_inner:
            logger.info(
                f"Skipping modality {k} for inconsistency check (cannot get batch size): {e_inner}"
            )

    if len(modality_batch_sizes) < 1:
        pytest.skip("Need at least one valid modality with retrievable batch size.")

    sorted_modalities = sorted(modality_batch_sizes.keys())
    reference_mod = sorted_modalities[0]
    bs1 = modality_batch_sizes[reference_mod]

    for mod_key in sorted_modalities[1:]:
        bs_other = modality_batch_sizes[mod_key]
        if bs1 != bs_other:
            skip_msg = (
                f"Fixture already has inconsistent batch sizes ({reference_mod}:{bs1}, "
                f"{mod_key}:{bs_other}). Cannot reliably test."
            )
            pytest.skip(skip_msg)

    if bs1 <= 1:
        pytest.skip(f"Batch size ({bs1}) too small to test inconsistency reliably.")

    mods_present = list(modality_batch_sizes.keys())
    mod_to_change = ""
    if "satellite_actual" in mods_present:
         mod_to_change = "satellite_actual"
    elif "pv" in mods_present:
        mod_to_change = "pv"
    elif "gsp" in mods_present:
         mod_to_change = "gsp"
    elif "solar_azimuth" in mods_present:
         mod_to_change = "solar_azimuth"
    elif "nwp" in mods_present:
         mod_to_change = "nwp"
    else:
         pytest.skip("Could not find a suitable modality to modify for batch size test.")

    new_bs = bs1 - 1
    logger.info(
        f"Testing batch size inconsistency check via expected_batch_size: "
        f"Modifying '{mod_to_change}' from {bs1} to {new_bs}, expecting {bs1}."
    )

    if mod_to_change == "nwp":
        if not isinstance(batch.get(mod_to_change), dict):
             pytest.skip("NWP not a dict in batch, cannot modify.")
        modified_nwp = False
        try:
            first_source = next(iter(batch[mod_to_change].keys()))
            first_source_dict = batch[mod_to_change][first_source]
            if not isinstance(first_source_dict, dict):
                 raise TypeError("Source value is not dict")
            array_key = next(
                k_inner for k_inner, v_inner in first_source_dict.items()
                if isinstance(v_inner, np.ndarray) and v_inner.ndim > 0 and v_inner.shape[0] == bs1
            )
            source_data = first_source_dict[array_key]
            new_shape = (new_bs,) + source_data.shape[1:]
            batch[mod_to_change][first_source][array_key] = np.zeros(
                new_shape, dtype=source_data.dtype
            )
            modified_nwp = True
            logger.info(f"Modified NWP source '{first_source}', key '{array_key}'")
        except (StopIteration, KeyError, AttributeError, IndexError, TypeError) as e_inner:
            logger.warning(f"Could not modify NWP for batch size test: {e_inner}")
            pytest.skip("Could not modify any NWP source for batch size test.")
        if not modified_nwp:
             pytest.skip("Failed to modify NWP source.")
    elif isinstance(batch.get(mod_to_change), np.ndarray):
        data_to_change = batch[mod_to_change]
        if data_to_change.ndim == 0 or data_to_change.shape[0] != bs1:
            skip_msg = (
                f"Cannot modify batch size for array {mod_to_change}, "
                f"shape {data_to_change.shape}"
            )
            pytest.skip(skip_msg)
        new_shape = (new_bs,) + data_to_change.shape[1:]
        batch[mod_to_change] = np.zeros(new_shape, dtype=data_to_change.dtype)
        logger.info(f"Modified non-NWP modality '{mod_to_change}'")
    else:
        fail_msg = (
            f"Selected modality '{mod_to_change}' "
            f"({type(batch.get(mod_to_change))}) is not testable."
        )
        pytest.fail(fail_msg)

    omega_config = OmegaConf.create(config_dict)
    omega_input_data_config = OmegaConf.create(valid_input_data_config)

    with pytest.raises(ValueError) as exc_info:
        validate(
            batch,
            omega_config,
            omega_input_data_config,
            expected_batch_size=bs1
        )

    error_message = str(exc_info.value)

    expected_message_pattern = (
        rf"Mismatch for '{mod_to_change}' in batch_size\. "
        rf"Expected size {bs1}, Got {new_bs}\."
    )
    assert re.search(expected_message_pattern, error_message), \
        f"Pattern <{expected_message_pattern}> not found in error: <{error_message}>"
    logger.info(f"Caught expected ValueError: {error_message}")
