"""
Validation module for PVNet.

This module provides functionality to validate input data against
model configuration requirements before feeding it to the model.
"""

import logging
from typing import Any, Dict, Union

import torch

logger = logging.getLogger(__name__)


def validate_sample(
    sample: Dict[str, Any], config: Dict[str, Any], raise_error: bool = False
) -> Dict[str, Union[bool, Dict[str, bool]]]:
    """
    Validates that a sample matches the expected dimensions and requirements
    specified in the model configuration.

    Args:
        sample: Dictionary containing input data tensors
        config: Model configuration dictionary
        raise_error: If True, raises ValueError on validation failure instead of returning results

    Returns:
        Dictionary with validation results containing:
            - 'valid': Overall validity (boolean)
            - 'details': Dict with specific validation results per component
    """
    results = {"valid": True, "details": {}}

    # Initialize details structure
    components = ["nwp", "satellite", "pv", "times", "dimensions"]
    for component in components:
        results["details"][component] = {"valid": True, "issues": []}

    # Validate NWP data
    if "nwp" in sample:
        for nwp_type, nwp_data in sample["nwp"].items():
            if nwp_type not in config.get("nwp_encoders_dict", {}):
                _add_issue(results, "nwp", f"Unknown NWP type: {nwp_type}")
                continue

            nwp_config = config["nwp_encoders_dict"][nwp_type]

            # Check channels
            expected_channels = nwp_config.get("in_channels")
            actual_channels = nwp_data.shape[1] if len(nwp_data.shape) > 1 else 0

            if expected_channels != actual_channels:
                _add_issue(
                    results,
                    "nwp",
                    f"{nwp_type} NWP has {actual_channels} channels, expected {expected_channels}",
                )

            # Check spatial dimensions
            expected_size = nwp_config.get("image_size_pixels")
            if expected_size:
                # Assuming shape format: [batch, channels, time, height, width]
                if len(nwp_data.shape) != 5:
                    _add_issue(
                        results,
                        "nwp",
                        f"{nwp_type} NWP has incorrect dimensions: {nwp_data.shape}, expected 5D tensor",
                    )
                else:
                    height, width = nwp_data.shape[3], nwp_data.shape[4]
                    if height != expected_size or width != expected_size:
                        _add_issue(
                            results,
                            "nwp",
                            f"{nwp_type} NWP has spatial dimensions {height}x{width}, expected {expected_size}x{expected_size}",
                        )

            # Check time dimension
            if "nwp_history_minutes" in config and "nwp_forecast_minutes" in config:
                expected_history = config["nwp_history_minutes"].get(nwp_type, 0)
                expected_forecast = config["nwp_forecast_minutes"].get(nwp_type, 0)
                nwp_interval = config.get("nwp_interval_minutes", {}).get(nwp_type, 60)

                # Calculate expected time steps
                expected_time_steps = (expected_history + expected_forecast) // nwp_interval

                # Check time dimension (3rd dimension, index 2)
                if len(nwp_data.shape) > 2:
                    time_steps = nwp_data.shape[2]
                    if time_steps != expected_time_steps:
                        _add_issue(
                            results,
                            "nwp",
                            f"{nwp_type} NWP has {time_steps} time steps, expected {expected_time_steps}",
                        )
    else:
        if config.get("nwp_encoders_dict"):
            _add_issue(results, "nwp", "NWP data missing but expected in configuration")

    # Validate satellite data
    if "satellite" in sample and config.get("sat_encoder"):
        sat_data = sample["satellite"]
        sat_config = config["sat_encoder"]

        # Check channels
        expected_channels = sat_config.get("in_channels")
        actual_channels = sat_data.shape[1] if len(sat_data.shape) > 1 else 0

        if expected_channels != actual_channels:
            _add_issue(
                results,
                "satellite",
                f"Satellite has {actual_channels} channels, expected {expected_channels}",
            )

        # Check spatial dimensions
        expected_size = sat_config.get("image_size_pixels")
        if expected_size:
            # Assuming shape format: [batch, channels, time, height, width]
            if len(sat_data.shape) != 5:
                _add_issue(
                    results,
                    "satellite",
                    f"Satellite has incorrect dimensions: {sat_data.shape}, expected 5D tensor",
                )
            else:
                height, width = sat_data.shape[3], sat_data.shape[4]
                if height != expected_size or width != expected_size:
                    _add_issue(
                        results,
                        "satellite",
                        f"Satellite has spatial dimensions {height}x{width}, expected {expected_size}x{expected_size}",
                    )

        # Check time dimension
        if "sat_history_minutes" in config:
            sat_history = config["sat_history_minutes"]
            # Assuming 15-minute intervals for satellite data (adjust if different)
            sat_interval = 15
            expected_time_steps = sat_history // sat_interval

            if len(sat_data.shape) > 2:
                time_steps = sat_data.shape[2]
                if time_steps != expected_time_steps:
                    _add_issue(
                        results,
                        "satellite",
                        f"Satellite has {time_steps} time steps, expected {expected_time_steps}",
                    )
    elif config.get("sat_encoder"):
        _add_issue(results, "satellite", "Satellite data missing but expected in configuration")

    # Validate PV site data
    if "pv" in sample and config.get("pv_encoder"):
        pv_data = sample["pv"]
        pv_config = config["pv_encoder"]

        # Check number of sites
        expected_sites = pv_config.get("num_sites")
        if expected_sites:
            # Assuming shape format: [batch, sites, features, time]
            if len(pv_data.shape) != 4:
                _add_issue(
                    results,
                    "pv",
                    f"PV data has incorrect dimensions: {pv_data.shape}, expected 4D tensor",
                )
            else:
                actual_sites = pv_data.shape[1]
                if actual_sites != expected_sites:
                    _add_issue(
                        results,
                        "pv",
                        f"PV data has {actual_sites} sites, expected {expected_sites}",
                    )

        # Check time dimension
        if "pv_history_minutes" in config:
            pv_history = config["pv_history_minutes"]
            # Assuming 30-minute intervals for PV data (adjust if different)
            pv_interval = 30
            expected_time_steps = pv_history // pv_interval

            if len(pv_data.shape) > 3:
                time_steps = pv_data.shape[3]
                if time_steps != expected_time_steps:
                    _add_issue(
                        results,
                        "pv",
                        f"PV data has {time_steps} time steps, expected {expected_time_steps}",
                    )
    elif config.get("pv_encoder"):
        _add_issue(results, "pv", "PV data missing but expected in configuration")

    # Validate sun features if required
    if config.get("include_sun", False):
        if "sun_features" not in sample:
            _add_issue(results, "times", "Sun features missing but required by configuration")

    # Validate any other specific requirements
    # (Add additional validation based on specific model requirements)

    # Check overall validity
    for component in components:
        if not results["details"][component]["valid"]:
            results["valid"] = False

    # If requested, raise error with detailed message on validation failure
    if raise_error and not results["valid"]:
        error_msg = "Sample validation failed:\n"
        for component, details in results["details"].items():
            if not details["valid"]:
                error_msg += f"  {component.upper()} issues:\n"
                for issue in details["issues"]:
                    error_msg += f"    - {issue}\n"
        raise ValueError(error_msg)

    return results


def _add_issue(results: Dict, component: str, issue: str):
    """Helper function to add an issue to the validation results"""
    results["details"][component]["valid"] = False
    results["details"][component]["issues"].append(issue)
    results["valid"] = False


def validate_batch(
    batch: Dict[str, Any], config: Dict[str, Any], raise_error: bool = False
) -> Dict[str, bool]:
    """
    Validates a batch of samples against model configuration.

    Args:
        batch: Dictionary of batched input tensors
        config: Model configuration dictionary
        raise_error: If True, raises ValueError on validation failure

    Returns:
        Dictionary with validation results
    """
    # For batched data, extract first sample for validation
    # This assumes all samples in batch have same dimensions
    sample = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            sample[key] = {k: v[0:1] if torch.is_tensor(v) else v for k, v in value.items()}
        elif torch.is_tensor(value):
            sample[key] = value[0:1]
        else:
            sample[key] = value

    return validate_sample(sample, config, raise_error)
