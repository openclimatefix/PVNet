from pvnet.models.base_model import BaseModel


def test_from_pretrained():
    model_name = "openclimatefix/pvnet_v2"
    model_version = "4203e12e719efd93da641c43d2e38527648f4915"

    _ = BaseModel.from_pretrained(
        model_name,
        revision=model_version,
    )