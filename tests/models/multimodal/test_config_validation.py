import pytest
import logging

from pvnet.models.multimodal.config_validation import validate_multimodal_config


def test_validate_multimodal_config_with_valid_dict(valid_config_dict):
    result = validate_multimodal_config(valid_config_dict)
    assert result == {"valid": True}


def test_validate_error_missing_required_top_level_key(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    del invalid_cfg["forecast_minutes"]
    with pytest.raises(KeyError, match=r"Configuration missing required key: 'forecast_minutes'"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_missing_required_section(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    del invalid_cfg["output_network"]
    with pytest.raises(KeyError, match=r"missing required section: 'output_network'"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_section_wrong_type(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    invalid_cfg["optimizer"] = "not-a-dictionary"
    with pytest.raises(TypeError, match=r"section 'optimizer'.*must be a dictionary"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_section_missing_target(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    invalid_cfg["output_network"] = {"some_other_key": 123}
    with pytest.raises(KeyError, match=r"section 'output_network'.*missing required sub-key: '_target_'"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_optional_section_wrong_type(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    invalid_cfg["sat_encoder"] = ["this", "is", "a", "list"]
    with pytest.raises(TypeError, match=r"section 'sat_encoder'.*must be a dictionary"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_optional_section_missing_target(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    invalid_cfg["pv_encoder"] = {"another_key": 456}
    with pytest.raises(KeyError, match=r"section 'pv_encoder'.*missing required sub-key: '_target_'"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_nwp_sub_item_missing_target(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    invalid_cfg["nwp_encoders_dict"] = invalid_cfg["nwp_encoders_dict"].copy()
    assert len(invalid_cfg["nwp_encoders_dict"]) > 0
    nwp_key = list(invalid_cfg["nwp_encoders_dict"].keys())[0]
    invalid_cfg["nwp_encoders_dict"][nwp_key] = {"wrong_key": 789}
    with pytest.raises(KeyError, match=rf"Source '{nwp_key}'.*missing required sub-key: \'_target_\'"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_missing_required_time_param_for_encoder(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
    assert "sat_encoder" in invalid_cfg and invalid_cfg["sat_encoder"]
    del invalid_cfg["sat_history_minutes"]
    with pytest.raises(KeyError, match=r"includes 'sat_encoder' but is missing 'sat_history_minutes'"):
        validate_multimodal_config(invalid_cfg)


def test_validate_error_nwp_time_keys_mismatch(valid_config_dict):
    invalid_cfg = valid_config_dict.copy()
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
    # Remove associated time param if present to isolate the empty section warning
    if "sat_history_minutes" in warn_cfg:
        del warn_cfg["sat_history_minutes"]
    with caplog.at_level(logging.WARNING):
        result = validate_multimodal_config(warn_cfg)
    assert result == {"valid": True}
    assert "Optional config section 'sat_encoder' is present but empty" in caplog.text

def test_validate_warning_missing_optional_sensor_time(valid_config_dict, caplog):
    warn_cfg = valid_config_dict.copy()
    warn_cfg["sensor_encoder"] = {"_target_": "some.SensorEncoder"}
    # Ensure optional time params are missing to test warnings
    if "sensor_history_minutes" in warn_cfg:
        del warn_cfg["sensor_history_minutes"]
    if "sensor_forecast_minutes" in warn_cfg:
        del warn_cfg["sensor_forecast_minutes"]
    with caplog.at_level(logging.WARNING):
        result = validate_multimodal_config(warn_cfg)
    assert result == {"valid": True}
    assert "includes 'sensor_encoder' but is missing 'sensor_history_minutes'" in caplog.text
    assert "includes 'sensor_encoder' but is missing 'sensor_forecast_minutes'" in caplog.text


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
    assert len(warn_cfg["nwp_history_minutes"]) > 0
    nwp_key = list(warn_cfg["nwp_history_minutes"].keys())[0]
    warn_cfg["nwp_history_minutes"][nwp_key] = 120.0
    with caplog.at_level(logging.WARNING):
        result = validate_multimodal_config(warn_cfg)
    assert result == {"valid": True}
    assert f"'nwp_history_minutes' for source '{nwp_key}' expected int, found float" in caplog.text
