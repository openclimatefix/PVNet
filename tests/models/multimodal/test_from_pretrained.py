from pvnet.models.base_model import BaseModel


def test_from_pretrained():
    model_name = "openclimatefix/pvnet_uk_region"
    model_version = "aa73cdafd1db8df3c8b7f5ecfdb160989e7639ac"

    _ = BaseModel.from_pretrained(
        model_name,
        revision=model_version,
    )
