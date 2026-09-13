def test_default_model_name(proxy, tiny_model):
    assert proxy.get_model_name() == type(tiny_model).__name__


def test_custom_model_name(tiny_model):
    from torch import optim, nn
    from mlinstruct.train.model_proxy import TorchModelProxy

    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    named_proxy = TorchModelProxy(
        model=tiny_model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        model_name="my-model",
    )

    assert named_proxy.get_model_name() == "my-model"
