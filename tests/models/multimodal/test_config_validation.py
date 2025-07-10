"""Tests for multimodal configuration validation functions."""

import logging
from copy import deepcopy

import pytest
from omegaconf import OmegaConf

from pvnet.models.multimodal.validation.config_validation import validate

logger = logging.getLogger(__name__)


def test_validate_valid_config(valid_config_dict):
    """Test validate with a valid config dictionary."""
    try:
        omega_multimodal_config = OmegaConf.create(valid_config_dict)
        validate(multimodal_config=omega_multimodal_config)
    except Exception as e:
        pytest.fail(f"validate raised an unexpected exception with a valid config: {e}")


@pytest.mark.parametrize(
    "key_to_delete, expected_error_match",
    [
        ("forecast_minutes", r"missing required key: 'forecast_minutes'"),
        ("output_network", r"missing required key: 'output_network'"),
    ],
    ids=[
        "missing_top_level_key",
        "missing_required_section",
    ],
)
def test_validate_static_error_missing_required_item(
    valid_config_dict, key_to_delete, expected_error_match
):
    """Tests validate catches static KeyError for missing required config items."""
    invalid_cfg_dict = deepcopy(valid_config_dict)

    if key_to_delete in invalid_cfg_dict:
        del invalid_cfg_dict[key_to_delete]
    else:
        pytest.skip(f"Key '{key_to_delete}' not in valid_config_dict fixture.")

    omega_invalid_cfg = OmegaConf.create(invalid_cfg_dict)

    with pytest.raises(KeyError, match=expected_error_match):
        validate(omega_invalid_cfg)


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
    valid_config_dict, section_name, invalid_value
):
    """Tests validate catches static TypeError for sections not being dicts."""
    invalid_cfg_dict = deepcopy(valid_config_dict)
    if section_name not in invalid_cfg_dict:
        pytest.skip(f"'{section_name}' not in fixture.")

    invalid_cfg_dict[section_name] = invalid_value
    match_str = rf"key '{section_name}' expected type <class 'dict'>"

    omega_invalid_cfg = OmegaConf.create(invalid_cfg_dict)

    with pytest.raises(TypeError, match=match_str):
        validate(omega_invalid_cfg)


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
    valid_config_dict, section_name, invalid_sub_dict
):
    """Tests validate catches static KeyError when section dict misses '_target_'."""
    invalid_cfg_dict = deepcopy(valid_config_dict)
    if section_name not in invalid_cfg_dict:
        pytest.skip(f"'{section_name}' not in fixture.")

    invalid_cfg_dict[section_name] = invalid_sub_dict
    match_str = rf"Section '{section_name}' missing required key: '_target_'"

    omega_invalid_cfg = OmegaConf.create(invalid_cfg_dict)

    with pytest.raises(KeyError, match=match_str):
        validate(omega_invalid_cfg)


def test_validate_static_error_nwp_sub_item_missing_target(valid_config_dict):
    """Tests validate catches static KeyError for NWP sub-item missing '_target_'."""
    invalid_cfg_dict = deepcopy(valid_config_dict)
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

    with pytest.raises(KeyError, match=match_str):
        validate(omega_invalid_cfg)


def test_validate_static_error_missing_req_time_param(valid_config_dict):
    """Tests validate catches static KeyError for missing dependent time param."""
    invalid_cfg_dict = deepcopy(valid_config_dict)
    if not invalid_cfg_dict.get("sat_encoder"):
        pytest.skip("sat_encoder not configured in fixture.")

    if "sat_history_minutes" in invalid_cfg_dict:
        del invalid_cfg_dict["sat_history_minutes"]
    else:
        pytest.skip("'sat_history_minutes' not in fixture.")

    match_str = r"Config with 'sat_encoder' missing required key: 'sat_history_minutes'"

    omega_invalid_cfg = OmegaConf.create(invalid_cfg_dict)

    with pytest.raises(KeyError, match=match_str):
        validate(omega_invalid_cfg)


def test_validate_static_error_nwp_time_keys_mismatch(valid_config_dict):
    """Tests validate catches static ValueError for NWP time key mismatch."""
    invalid_cfg_dict = deepcopy(valid_config_dict)
    if not invalid_cfg_dict.get("nwp_encoders_dict") or not invalid_cfg_dict.get(
        "nwp_history_minutes"
    ):
        pytest.skip("NWP sections missing or empty in fixture.")

    invalid_cfg_dict["nwp_history_minutes"] = deepcopy(
        invalid_cfg_dict["nwp_history_minutes"]
    )
    if "dummy_source_for_mismatch" not in invalid_cfg_dict["nwp_history_minutes"]:
        invalid_cfg_dict["nwp_history_minutes"]["dummy_source_for_mismatch"] = 60
    else:
        pytest.skip("Cannot reliably create mismatch key.")

    match_str = r"Keys in 'nwp_history_minutes'.*do not match sources"

    omega_invalid_cfg = OmegaConf.create(invalid_cfg_dict)

    with pytest.raises(ValueError, match=match_str):
        validate(omega_invalid_cfg)
