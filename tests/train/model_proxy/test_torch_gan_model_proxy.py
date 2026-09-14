import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn, optim  # noqa: E402

from mlinstruct.train.model_proxy import GANModelProxy, OnnxExportable  # noqa: E402
from mlinstruct.train.model_proxy.torch_gan_model_proxy import (  # noqa: E402
    bce_discriminator_loss,
    non_saturating_generator_loss,
)
from mlinstruct.utils.exception import ModelProxyError  # noqa: E402

onnxruntime = pytest.importorskip("onnxruntime")


def test_train_one_batch_returns_scalar_losses(gan_proxy, real_samples_loader):
    real_batch = next(iter(real_samples_loader))[0]

    g_loss, d_loss = gan_proxy.train_one_batch(real_batch)

    assert isinstance(g_loss, float)
    assert isinstance(d_loss, float)


def test_discriminator_steps_before_generator_step(gan_proxy, real_samples_loader, mocker):
    real_batch = next(iter(real_samples_loader))[0]

    manager = mocker.Mock()
    manager.attach_mock(mocker.spy(gan_proxy._discriminator_optimizer, "step"), "d_step")
    manager.attach_mock(mocker.spy(gan_proxy._generator_optimizer, "step"), "g_step")

    gan_proxy.train_one_batch(real_batch)

    call_names = [call[0] for call in manager.mock_calls]
    assert call_names == ["d_step", "g_step"]


def test_n_critic_runs_multiple_discriminator_steps(gan_proxy, real_samples_loader, mocker):
    real_batch = next(iter(real_samples_loader))[0]
    d_step_spy = mocker.spy(gan_proxy._discriminator_optimizer, "step")
    g_step_spy = mocker.spy(gan_proxy._generator_optimizer, "step")

    gan_proxy.train_one_batch(real_batch, n_critic=3)

    assert d_step_spy.call_count == 3
    assert g_step_spy.call_count == 1


def test_generate_returns_requested_sample_count(gan_proxy):
    samples = gan_proxy.generate(5)

    assert samples.shape[0] == 5


def test_generate_output_requires_no_grad(gan_proxy):
    samples = gan_proxy.generate(4)

    assert samples.requires_grad is False


def test_save_then_load_roundtrip(gan_proxy, real_samples_loader, save_dir):
    real_batch = next(iter(real_samples_loader))[0]
    gan_proxy.train_one_batch(real_batch)

    original_g_state = {k: v.clone() for k, v in gan_proxy._generator.state_dict().items()}
    original_d_state = {k: v.clone() for k, v in gan_proxy._discriminator.state_dict().items()}
    original_g_lr = gan_proxy._generator_optimizer.state_dict()["param_groups"][0]["lr"]
    original_d_lr = gan_proxy._discriminator_optimizer.state_dict()["param_groups"][0]["lr"]

    gan_proxy.save_weights(
        epoch=2, save_dir_path=save_dir, model_name="gan", g_loss=0.1, d_loss=0.2
    )

    with torch.no_grad():
        for param in gan_proxy._generator.parameters():
            param.zero_()
        for param in gan_proxy._discriminator.parameters():
            param.zero_()

    loaded_epoch = gan_proxy.load_checkpoint(save_dir / "gan.pt")

    assert loaded_epoch == 2
    for key, value in original_g_state.items():
        assert torch.equal(gan_proxy._generator.state_dict()[key], value)
    for key, value in original_d_state.items():
        assert torch.equal(gan_proxy._discriminator.state_dict()[key], value)
    assert gan_proxy._generator_optimizer.state_dict()["param_groups"][0]["lr"] == original_g_lr
    assert gan_proxy._discriminator_optimizer.state_dict()["param_groups"][0]["lr"] == original_d_lr


def test_save_weights_rejects_missing_dir(gan_proxy, save_dir):
    missing_dir = save_dir / "does_not_exist"

    with pytest.raises(ModelProxyError):
        gan_proxy.save_weights(
            epoch=1, save_dir_path=missing_dir, model_name="gan", g_loss=0.1, d_loss=0.2
        )


def test_get_model_name_default_and_custom(tiny_generator, tiny_discriminator):
    default_proxy = GANModelProxy(
        generator=tiny_generator,
        discriminator=tiny_discriminator,
        generator_optimizer=optim.SGD(tiny_generator.parameters(), lr=0.01),
        discriminator_optimizer=optim.SGD(tiny_discriminator.parameters(), lr=0.01),
        latent_dim=3,
    )
    assert default_proxy.get_model_name() == type(tiny_generator).__name__

    named_proxy = GANModelProxy(
        generator=tiny_generator,
        discriminator=tiny_discriminator,
        generator_optimizer=optim.SGD(tiny_generator.parameters(), lr=0.01),
        discriminator_optimizer=optim.SGD(tiny_discriminator.parameters(), lr=0.01),
        latent_dim=3,
        model_name="my-gan",
    )
    assert named_proxy.get_model_name() == "my-gan"


def test_default_generator_loss_is_non_saturating_bce():
    d_pred_fake = torch.tensor([0.3, 0.7])

    loss = non_saturating_generator_loss(d_pred_fake)

    expected = nn.functional.binary_cross_entropy(d_pred_fake, torch.ones_like(d_pred_fake))
    assert torch.isclose(loss, expected)


def test_default_discriminator_loss_is_bce():
    d_pred_real = torch.tensor([0.9, 0.6])
    d_pred_fake = torch.tensor([0.2, 0.4])

    loss = bce_discriminator_loss(d_pred_real, d_pred_fake)

    expected = nn.functional.binary_cross_entropy(
        d_pred_real, torch.ones_like(d_pred_real)
    ) + nn.functional.binary_cross_entropy(d_pred_fake, torch.zeros_like(d_pred_fake))
    assert torch.isclose(loss, expected)


def test_custom_loss_fns_are_used(tiny_generator, tiny_discriminator, real_samples_loader, mocker):
    g_loss_fn = mocker.Mock(side_effect=non_saturating_generator_loss)
    d_loss_fn = mocker.Mock(side_effect=bce_discriminator_loss)
    proxy = GANModelProxy(
        generator=tiny_generator,
        discriminator=tiny_discriminator,
        generator_optimizer=optim.SGD(tiny_generator.parameters(), lr=0.01),
        discriminator_optimizer=optim.SGD(tiny_discriminator.parameters(), lr=0.01),
        latent_dim=3,
        generator_loss_fn=g_loss_fn,
        discriminator_loss_fn=d_loss_fn,
    )
    real_batch = next(iter(real_samples_loader))[0]

    proxy.train_one_batch(real_batch)

    g_loss_fn.assert_called_once()
    d_loss_fn.assert_called_once()


def test_gan_model_proxy_is_onnx_exportable(gan_proxy):
    assert isinstance(gan_proxy, OnnxExportable)


def test_export_onnx_exports_generator_only(gan_proxy, save_dir):
    onnx_path = save_dir / "generator.onnx"
    noise = gan_proxy.sample_noise(4)

    gan_proxy.export_onnx(onnx_path, noise)

    assert onnx_path.exists()

    gan_proxy._generator.eval()
    with torch.no_grad():
        expected_output = gan_proxy._generator(noise).numpy()

    session = onnxruntime.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    (actual_output,) = session.run(None, {input_name: noise.numpy()})

    assert np.allclose(actual_output, expected_output, atol=1e-5)
