import pytest
import logging
import numpy as np
from copy import deepcopy

NumpyBatch = dict

from pvnet.models.multimodal.config_validation import validate

logger = logging.getLogger(__name__)


def test_validate_valid_inputs(valid_config_dict, sample_numpy_batch):
    """Test validate with valid config and correctly shaped batch."""
    try:
        validate(sample_numpy_batch, valid_config_dict)
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
    ]
)
def test_validate_static_error_missing_required_item(valid_config_dict, key_to_delete, expected_error_match):
    """Tests validate catches static KeyError for missing required config items."""
    invalid_cfg = deepcopy(valid_config_dict)
    dummy_batch = {}

    if key_to_delete in invalid_cfg:
        del invalid_cfg[key_to_delete]
    else:
        pytest.skip(f"Key '{key_to_delete}' not in valid_config_dict fixture.")

    with pytest.raises(KeyError, match=expected_error_match):
        validate(dummy_batch, invalid_cfg)


@pytest.mark.parametrize(
    "section_name, invalid_value",
    [
        ("optimizer", "not-a-dictionary"),
        ("sat_encoder", ["list", "not", "dict"]),
    ],
    ids=[
        "required_section_wrong_type",
        "optional_section_wrong_type",
    ]
)
def test_validate_static_error_section_wrong_type(valid_config_dict, section_name, invalid_value):
    """Tests validate catches static TypeError for sections not being dicts."""
    invalid_cfg = deepcopy(valid_config_dict)
    dummy_batch = {}
    if section_name not in invalid_cfg: pytest.skip(f"'{section_name}' not in fixture.")

    invalid_cfg[section_name] = invalid_value
    with pytest.raises(TypeError, match=rf"section '{section_name}'.*must be a dictionary"):
        validate(dummy_batch, invalid_cfg)


@pytest.mark.parametrize(
    "section_name, invalid_sub_dict",
    [
        ("output_network", {"some_other_key": 123}),
        ("pv_encoder", {"another_key": 456}),
    ],
    ids=[
        "required_section_missing_target",
        "optional_section_missing_target",
    ]
)
def test_validate_static_error_section_missing_target(valid_config_dict, section_name, invalid_sub_dict):
    """Tests validate catches static KeyError when section dict misses '_target_'."""
    invalid_cfg = deepcopy(valid_config_dict)
    dummy_batch = {}
    if section_name not in invalid_cfg: pytest.skip(f"'{section_name}' not in fixture.")

    invalid_cfg[section_name] = invalid_sub_dict
    with pytest.raises(KeyError, match=rf"section '{section_name}'.*missing required sub-key: '_target_'"):
        validate(dummy_batch, invalid_cfg)


def test_validate_static_error_nwp_sub_item_missing_target(valid_config_dict):
    """Tests validate catches static KeyError for NWP sub-item missing '_target_'."""
    invalid_cfg = deepcopy(valid_config_dict)
    dummy_batch = {}
    if not invalid_cfg.get("nwp_encoders_dict"):
        pytest.skip("nwp_encoders_dict missing or empty in fixture.")

    invalid_cfg["nwp_encoders_dict"] = deepcopy(invalid_cfg["nwp_encoders_dict"])
    # Use try-except to handle potentially empty dict after deepcopy if original was empty
    try:
        nwp_key = list(invalid_cfg["nwp_encoders_dict"].keys())[0]
        invalid_cfg["nwp_encoders_dict"][nwp_key] = {"wrong_key": 789}
    except IndexError:
        pytest.skip("nwp_encoders_dict is empty in fixture.")


    with pytest.raises(KeyError, match=rf"Source '{nwp_key}'.*missing required sub-key: '_target_'"):
        validate(dummy_batch, invalid_cfg)


def test_validate_static_error_missing_req_time_param(valid_config_dict):
    """Tests validate catches static KeyError for missing dependent time param."""
    invalid_cfg = deepcopy(valid_config_dict)
    dummy_batch = {}
    if not invalid_cfg.get("sat_encoder"):
         pytest.skip("sat_encoder not configured in fixture.")

    if "sat_history_minutes" in invalid_cfg:
        del invalid_cfg["sat_history_minutes"]
    else:
         pytest.skip("'sat_history_minutes' not in fixture.")

    with pytest.raises(KeyError, match=r"includes 'sat_encoder' but is missing 'sat_history_minutes'"):
        validate(dummy_batch, invalid_cfg)


def test_validate_static_error_nwp_time_keys_mismatch(valid_config_dict):
    """Tests validate catches static ValueError for NWP time key mismatch."""
    invalid_cfg = deepcopy(valid_config_dict)
    dummy_batch = {}
    if not invalid_cfg.get("nwp_encoders_dict") or \
       not invalid_cfg.get("nwp_history_minutes"):
        pytest.skip("NWP sections missing or empty in fixture.")

    invalid_cfg["nwp_history_minutes"] = deepcopy(invalid_cfg["nwp_history_minutes"])
    if "dummy_source_for_mismatch" not in invalid_cfg["nwp_history_minutes"]:
        invalid_cfg["nwp_history_minutes"]["dummy_source_for_mismatch"] = 60
    else:
        pytest.skip("Cannot reliably create mismatch key.")

    with pytest.raises(ValueError, match=r"Keys in 'nwp_history_minutes'.*do not match sources"):
        validate(dummy_batch, invalid_cfg)


def test_validate_batch_error_missing_modality_key(valid_config_dict, sample_numpy_batch):
    """Test KeyError when a required modality is missing from the batch."""
    config = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)
    if not config.get("sat_encoder"): pytest.skip("sat_encoder not in config fixture.")

    if "satellite_actual" in batch:
        del batch["satellite_actual"]
    elif config.get("sat_encoder"):
         # If sat is configured but not in sample_numpy_batch, the fixture/data is inconsistent
         pytest.skip("Config expects satellite_actual, but sample_numpy_batch missing it.")

    with pytest.raises(KeyError, match="Batch missing 'satellite_actual' data"):
        validate(batch, config)


def test_validate_batch_error_modality_wrong_type(valid_config_dict, sample_numpy_batch):
    """Test TypeError when batch data for a modality is not a numpy array."""
    config = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)
    if not config.get("sat_encoder"): pytest.skip("sat_encoder not in config fixture.")
    if "satellite_actual" not in batch and config.get("sat_encoder"):
        pytest.skip("Config expects satellite_actual, but sample_numpy_batch missing it.")
    elif "satellite_actual" not in batch:
         pytest.skip("satellite_actual not in batch, cannot test wrong type.")


    batch["satellite_actual"] = "this is not a numpy array"

    with pytest.raises(TypeError, match="'satellite_actual' data must be np.ndarray"):
        validate(batch, config)


def test_validate_batch_error_wrong_ndim(valid_config_dict, sample_numpy_batch):
    """Test ValueError for incorrect number of dimensions."""
    config = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)
    if not config.get("sat_encoder"): pytest.skip("sat_encoder not in config fixture.")
    if "satellite_actual" not in batch and config.get("sat_encoder"):
        pytest.skip("Config expects satellite_actual, but sample_numpy_batch missing it.")
    elif "satellite_actual" not in batch:
         pytest.skip("satellite_actual not in batch, cannot test wrong ndim.")


    correct_shape = batch["satellite_actual"].shape
    if len(correct_shape) <= 1: pytest.skip("Cannot reduce ndim further.")
    batch["satellite_actual"] = np.zeros(correct_shape[1:]) # Missing batch dim

    with pytest.raises(ValueError, match=r"'satellite_actual' data expected 5 dimensions.*|shape error.*Expected B x"):
        validate(batch, config)


def test_validate_batch_error_wrong_shape_time(valid_config_dict, sample_numpy_batch):
    """Test ValueError for incorrect time dimension size."""
    config = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)
    if not config.get("sat_encoder"): pytest.skip("sat_encoder not in config fixture.")
    if "satellite_actual" not in batch and config.get("sat_encoder"):
        pytest.skip("Config expects satellite_actual, but sample_numpy_batch missing it.")
    elif "satellite_actual" not in batch:
         pytest.skip("satellite_actual not in batch, cannot test wrong time shape.")


    correct_shape = batch["satellite_actual"].shape
    if len(correct_shape) < 2: pytest.skip("Batch has insufficient dimensions.")
    wrong_time_shape = list(correct_shape)
    wrong_time_shape[1] += 1 # Add one extra time step
    batch["satellite_actual"] = np.zeros(tuple(wrong_time_shape))

    with pytest.raises(ValueError, match=rf"'{'satellite_actual'}' shape error using interval .* Got \({batch['satellite_actual'].shape}\)"):
        validate(batch, config)


def test_validate_batch_error_wrong_shape_spatial(valid_config_dict, sample_numpy_batch):
    """Test ValueError for incorrect spatial dimension size."""
    config = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)
    if not config.get("sat_encoder"): pytest.skip("sat_encoder not in config fixture.")
    if "satellite_actual" not in batch and config.get("sat_encoder"):
        pytest.skip("Config expects satellite_actual, but sample_numpy_batch missing it.")
    elif "satellite_actual" not in batch:
         pytest.skip("satellite_actual not in batch, cannot test wrong spatial shape.")


    correct_shape = batch["satellite_actual"].shape
    if len(correct_shape) < 3: pytest.skip("Batch has insufficient dimensions.")
    wrong_spatial_shape = list(correct_shape)
    wrong_spatial_shape[2] += 1 # Add one pixel to height
    batch["satellite_actual"] = np.zeros(tuple(wrong_spatial_shape))

    with pytest.raises(ValueError, match=rf"'{'satellite_actual'}' shape error using interval .* Got \({batch['satellite_actual'].shape}\)"):
        validate(batch, config)


def test_validate_batch_error_batch_size_inconsistency(valid_config_dict, sample_numpy_batch):
    """Test ValueError for inconsistent batch sizes across modalities."""
    config = deepcopy(valid_config_dict)
    batch = deepcopy(sample_numpy_batch)

    def get_batch_size(batch_dict, key):
        """Helper to get batch size, handling NWP dict structure."""
        if key not in batch_dict:
            raise KeyError(f"Key '{key}' not found in batch for batch size check.")
        data = batch_dict[key]

        if key == "nwp":
            if not isinstance(data, dict) or not data:
                 raise TypeError(f"NWP data for key '{key}' in batch is not a non-empty dict")
            first_source_key = next(iter(data.keys()))
            first_source_dict = data[first_source_key]
            if not isinstance(first_source_dict, dict):
                raise TypeError(f"NWP source value for '{first_source_key}' is not a dict: {first_source_dict}")
            try:
                data_array = next(v for v in first_source_dict.values() if isinstance(v, np.ndarray) and v.ndim > 0)
            except StopIteration:
                raise ValueError(f"No numpy array with ndim > 0 found within NWP source dict: {first_source_dict}")
            return data_array.shape[0]
        elif isinstance(data, np.ndarray):
            if data.ndim == 0:
                 raise ValueError(f"Data for key '{key}' is a 0-dim array, cannot get batch size.")
            return data.shape[0]
        else:
            raise TypeError(f"Data for key '{key}' ({type(data)}) is not a numpy array or NWP dict.")

    modalities_in_batch = [
        k for k in ["satellite_actual", "pv", "nwp", "gsp", "sun"]
        if k in batch and isinstance(batch[k], (np.ndarray, dict))
    ]

    valid_modalities = []
    for k in modalities_in_batch:
        try:
            get_batch_size(batch, k)
            valid_modalities.append(k)
        except (TypeError, ValueError, KeyError, StopIteration) as e:
             logger.info(f"Skipping modality {k} for inconsistency check (cannot get batch size): {e}")

    if len(valid_modalities) < 2:
         pytest.skip(f"Need at least two valid modalities ({valid_modalities}) with retrievable batch sizes to test inconsistency.")

    mod1_key = valid_modalities[0]
    mod2_key = valid_modalities[1]

    try:
        bs1 = get_batch_size(batch, mod1_key)
        bs2 = get_batch_size(batch, mod2_key)
    except (KeyError, TypeError, ValueError) as e:
         pytest.fail(f"Failed to get initial batch sizes for {mod1_key}/{mod2_key}: {e}")

    if bs1 != bs2: pytest.skip(f"Fixture already has inconsistent batch sizes ({mod1_key}:{bs1}, {mod2_key}:{bs2}).")
    if bs1 <= 1: pytest.skip(f"Batch size ({bs1}) too small to test inconsistency reliably.")

    new_bs = bs1 - 1
    mod_to_change = mod2_key

    if mod_to_change == "nwp":
         if not isinstance(batch[mod_to_change], dict): pytest.skip("NWP not a dict, cannot modify.")
         for source in batch[mod_to_change]:
             if not isinstance(batch[mod_to_change][source], dict): continue
             try:
                   array_key = next(k for k, v in batch[mod_to_change][source].items() if isinstance(v, np.ndarray) and v.ndim > 0)
                   source_data = batch[mod_to_change][source][array_key]
                   if source_data.ndim == 0: continue
                   new_shape = (new_bs,) + source_data.shape[1:]
                   batch[mod_to_change][source][array_key] = np.zeros(new_shape, dtype=source_data.dtype)
             except (StopIteration, KeyError, AttributeError, IndexError):
                   logger.warning(f"Could not find/modify array in NWP source '{source}' for batch size test.")
                   pytest.skip(f"Could not modify NWP source {source} for batch size test.")
    elif isinstance(batch[mod_to_change], np.ndarray):
        data_to_change = batch[mod_to_change]
        if data_to_change.ndim == 0: pytest.skip(f"Cannot modify batch size for 0-dim array {mod_to_change}")
        new_shape = (new_bs,) + data_to_change.shape[1:]
        batch[mod_to_change] = np.zeros(new_shape, dtype=data_to_change.dtype)
    else:
         pytest.fail(f"Modality '{mod_to_change}' to change is not an ndarray or NWP dict.")

    expected_match = rf"Batch size mismatch for {mod_to_change}.*: expected {bs1}, got {new_bs}"
    with pytest.raises(ValueError, match=expected_match):
        validate(batch, config)
