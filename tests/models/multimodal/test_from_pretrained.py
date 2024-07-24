from pvnet.models.base_model import BaseModel


def test_from_pretrained():
    model_name = "openclimatefix/pvnet_uk_region"
    model_version = "92266cd9040c590a9e90ee33eafd0e7b92548be8"

    _ = BaseModel.from_pretrained(
        model_id=model_name,
        revision=model_version,
    )
