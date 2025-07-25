from torch.optim import SGD


def test_model_forward(late_fusion_model, sample_batch):
    y = late_fusion_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape) == (2, 16), y.shape

def test_model_forward_site_history(late_fusion_model_site_history, sample_site_batch):

    y = late_fusion_model_site_history(sample_site_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape) == (2, 16), y.shape


def test_model_backward(late_fusion_model, sample_batch):
    opt = SGD(late_fusion_model.parameters(), lr=0.001)

    y = late_fusion_model(sample_batch)

    # Backwards on sum drives sum to zero
    y.sum().backward()


def test_quantile_model_forward(late_fusion_quantile_model, sample_batch):
    y_quantiles = late_fusion_quantile_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15, num_quantiles=3
    assert tuple(y_quantiles.shape) == (2, 16, 3), y_quantiles.shape


def test_quantile_model_backward(late_fusion_quantile_model, sample_batch):
    opt = SGD(late_fusion_quantile_model.parameters(), lr=0.001)

    y_quantiles = late_fusion_quantile_model(sample_batch)

    # Backwards on sum drives sum to zero
    y_quantiles.sum().backward()
