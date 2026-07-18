import torch

from kkbox.models import MultiTaskFMNet


def _toy_model():
    cat_cols = ["city_enc", "gender_enc"]
    cardinalities = {"city_enc": 5, "gender_enc": 3}
    embed_dims = {"city_enc": 2, "gender_enc": 2}
    num_numerical = 4
    return MultiTaskFMNet(
        cat_cols, cardinalities, embed_dims, num_numerical,
        fm_k=3, backbone_dims=(8, 4), dropouts=(0.1, 0.1), head_hidden_dim=4,
    )


def test_forward_output_shapes():
    model = _toy_model()
    batch_size = 16
    x_num = torch.randn(batch_size, 4)
    x_cat = torch.stack([torch.randint(0, 5, (batch_size,)), torch.randint(0, 3, (batch_size,))], dim=1)
    churn_logit, ltv_pred = model(x_num, x_cat)
    assert churn_logit.shape == (batch_size,)
    assert ltv_pred.shape == (batch_size,)


def test_backward_populates_gradients_on_every_parameter():
    model = _toy_model()
    x_num = torch.randn(8, 4)
    x_cat = torch.stack([torch.randint(0, 5, (8,)), torch.randint(0, 3, (8,))], dim=1)
    churn_logit, ltv_pred = model(x_num, x_cat)
    loss = churn_logit.mean() + ltv_pred.mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"


def test_unknown_category_index_does_not_crash_embedding():
    # cardinality includes the reserved __unknown__ bucket as the last index
    model = _toy_model()
    x_num = torch.randn(2, 4)
    x_cat = torch.tensor([[4, 2], [4, 2]])  # last valid index for each (the unknown bucket)
    churn_logit, ltv_pred = model(x_num, x_cat)
    assert torch.isfinite(churn_logit).all()
    assert torch.isfinite(ltv_pred).all()
