import pytest
import logging

from pvnet.models.multimodal.config_validation import validate_multimodal_config


def test_validate_multimodal_config_with_valid_dict(valid_config_dict):
    result = validate_multimodal_config(valid_config_dict)
    assert result == {"valid": True}


@pytest.mark.parametrize(
    "key_to_delete, expected_error_match",
    [
        ("forecast_minutes", r"Configuration missing required key: 'forecast_minutes'"),
        ("output_network", r"missing required section: 'output_network'"),
    ],
    ids=[
        "missing_top_level_key",
        "missing_required_section",
    ]
)


def test_validate_error_missing_required_item(valid_config_dict, key_to_delete, expected_error_match):
    """Tests KeyError when required keys or sections are missing."""
    invalid_cfg = valid_config_dict.copy()
    if key_to_delete in invalid_cfg:
        del invalid_cfg[key_to_delete]
    else:
        pytest.skip(f"Key '{key_to_delete}' not found in valid_config_dict fixture.")

    with pytest.raises(KeyError, match=expected_error_match):
        validate_multimodal_config(invalid_cfg)


@pytest.mark.parametrize(
    "section_name, invalid_value",
    [
        ("optimizer", "not-a-dictionary"),
        ("sat_encoder", ["this", "is", "a", "list"]),
    ],
    ids=[
        "required_section_wrong_type",
        "optional_section_wrong_type",
    ]
)


def test_validate_error_section_wrong_type_parametrized(valid_config_dict, section_name, invalid_value):
    """Tests TypeError when a section is not a dictionary."""
    invalid_cfg = valid_config_dict.copy()
    invalid_cfg[section_name] = invalid_value
    with pytest.raises(TypeError, match=rf"section '{section_name}'.*must be a dictionary"):
        validate_multimodal_config(invalid_cfg)


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


def test_validate_error_section_missing_target_parametrized(valid_config_dict, section_name, invalid_sub_dict):
    """Tests KeyError when a section dictionary is missing the '_target_' sub-key."""
    invalid_cfg = valid_config_dict.copy()
    invalid_cfg[section_name] = invalid_sub_dict
    with pytest.raises(KeyError, match=rf"section '{section_name}'.*missing required sub-key: '_target_'"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_nwp_sub_item_missing_target(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    invalid_cfg["nwp_encoders_dict"] = invalid_cfg["nwp_encoders_dict"].copy()
    # Ensure NWP dict isn't empty in the fixture
    if not invalid_cfg["nwp_encoders_dict"]:
         pytest.skip("nwp_encoders_dict is empty in valid_config_dict fixture.")
    assert len(invalid_cfg["nwp_encoders_dict"]) > 0
    nwp_key = list(invalid_cfg["nwp_encoders_dict"].keys())[0]
    invalid_cfg["nwp_encoders_dict"][nwp_key] = {"wrong_key": 789}
    with pytest.raises(KeyError, match=rf"Source '{nwp_key}'.*missing required sub-key: \'_target_\'"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_missing_required_time_param_for_encoder(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    # Ensure sat_encoder exists for this test
    if "sat_encoder" not in invalid_cfg or not invalid_cfg["sat_encoder"]:
         pytest.skip("sat_encoder not configured in valid_config_dict fixture for this test.")
    assert "sat_encoder" in invalid_cfg and invalid_cfg["sat_encoder"]
    del invalid_cfg["sat_history_minutes"]
    with pytest.raises(KeyError, match=r"includes 'sat_encoder' but is missing 'sat_history_minutes'"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_nwp_time_keys_mismatch(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    # Ensure NWP time dicts exist and are non-empty
    if "nwp_history_minutes" not in invalid_cfg or not invalid_cfg["nwp_history_minutes"]:
        pytest.skip("nwp_history_minutes not configured or empty in valid_config_dict fixture.")
    assert len(invalid_cfg["nwp_history_minutes"]) > 0
    nwp_key_to_remove = list(invalid_cfg["nwp_history_minutes"].keys())[0]
    del invalid_cfg["nwp_history_minutes"][nwp_key_to_remove]
    with pytest.raises(ValueError, match=r"Keys in 'nwp_history_minutes'.*do not match sources"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_nwp_time_param_wrong_type(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    invalid_cfg["nwp_forecast_minutes"] = ["list", "is", "wrong", "type"]
    with pytest.raises(TypeError, match=r"'nwp_forecast_minutes'.*expected type dict"):
        validate_multimodal_config(invalid_cfg)


def test_validate_warning_non_int_time_param(valid_config_dict, caplog):
    warn_cfg = valid_config_dict.copy()
    warn_cfg["forecast_minutes"] = 60.5
    with caplog.at_level(logging.WARNING):
        result = validate_multimodal_config(warn_cfg)
    assert result == {"valid": True}
    assert "key 'forecast_minutes' expected type int, found float" in caplog.text


def test_validate_warning_empty_optional_section(valid_config_dict, caplog):
    warn_cfg = valid_config_dict.copy()
    warn_cfg["sat_encoder"] = {}
    # Remove associated time param if present to isolate empty section warning
    if "sat_history_minutes" in warn_cfg:
        del warn_cfg["sat_history_minutes"]
    with caplog.at_level(logging.WARNING):
        result = validate_multimodal_config(warn_cfg)
    assert result == {"valid": True}
    assert "Optional config section 'sat_encoder' is present but empty" in caplog.text


def test_validate_warning_nwp_empty_sources(valid_config_dict, caplog):
    warn_cfg = valid_config_dict.copy()
    warn_cfg["nwp_encoders_dict"] = {}
    # Remove time params if present to isolate the nwp source warning
    if "nwp_history_minutes" in warn_cfg: del warn_cfg["nwp_history_minutes"]
    if "nwp_forecast_minutes" in warn_cfg: del warn_cfg["nwp_forecast_minutes"]
    with caplog.at_level(logging.WARNING):
        result = validate_multimodal_config(warn_cfg)
    assert result == {"valid": True}
    assert "Optional config section 'nwp_encoders_dict' is present but empty" in caplog.text
    assert "'nwp_encoders_dict' is defined but contains no NWP sources" in caplog.text


def test_validate_warning_nwp_non_int_time_value(valid_config_dict, caplog):
    warn_cfg = valid_config_dict.copy()
    # Ensure NWP time dicts exist and are non-empty
    if "nwp_history_minutes" not in warn_cfg or not warn_cfg["nwp_history_minutes"]:
        pytest.skip("nwp_history_minutes not configured or empty in valid_config_dict fixture.")
    assert len(warn_cfg["nwp_history_minutes"]) > 0
    nwp_key = list(warn_cfg["nwp_history_minutes"].keys())[0]
    warn_cfg["nwp_history_minutes"][nwp_key] = 120.0
    with caplog.at_level(logging.WARNING):
        result = validate_multimodal_config(warn_cfg)
    assert result == {"valid": True}
    assert f"'nwp_history_minutes' for source '{nwp_key}' expected int, found float" in caplog.text
