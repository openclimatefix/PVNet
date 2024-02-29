from pvnet.models.ensemble import Ensemble


def test_model_init(multimodal_model):
    ensemble_model = Ensemble(
        model_list=[multimodal_model] * 3,
        weights=None,
    )

    ensemble_model = Ensemble(
        model_list=[multimodal_model] * 3,
        weights=[1, 2, 3],
    )


def test_model_forward(multimodal_model, sample_batch):
    ensemble_model = Ensemble(
        model_list=[multimodal_model] * 3,
    )

    y = ensemble_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape) == (2, 16), y.shape


def test_quantile_model_forward(multimodal_quantile_model, sample_batch):
    ensemble_model = Ensemble(
        model_list=[multimodal_quantile_model] * 3,
    )

    y_quantiles = ensemble_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15, num_quantiles=3
    assert tuple(y_quantiles.shape) == (2, 16, 3), y_quantiles.shape
