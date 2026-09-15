"""Vanilla GAN example: GANModelProxy + GANTrainer learn to generate samples
from two Gaussian blobs, alternating n_critic discriminator steps with one
generator step per batch.

Run with: uv run python examples/gan_training.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_blobs
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from mlinstruct.train.model_proxy import GANModelProxy
from mlinstruct.train.trainer import GANTrainer


def main() -> None:
    torch.manual_seed(0)

    X, _ = make_blobs(n_samples=512, centers=2, cluster_std=0.6, random_state=0)
    real_samples = torch.tensor(X, dtype=torch.float32)
    real_loader = DataLoader(TensorDataset(real_samples), batch_size=32, shuffle=True)

    latent_dim = 8
    generator = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 2))
    discriminator = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    gan_proxy = GANModelProxy(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=optim.Adam(generator.parameters(), lr=2e-4),
        discriminator_optimizer=optim.Adam(discriminator.parameters(), lr=2e-4),
        latent_dim=latent_dim,
    )

    gan_trainer = GANTrainer(
        model_proxy=gan_proxy,
        train_data=real_loader,
        n_critic=1,
        save_dir_path=Path("examples_output"),
    )

    result = gan_trainer.train(max_epochs=100)
    print(f"g_loss={result.g_loss_list[-1]:.4f} d_loss={result.d_loss_list[-1]:.4f}")
    print(f"Checkpoints under: {result.model_save_path}")

    fake_samples = gan_proxy.generate(n_samples=256).cpu().numpy()

    fig, (loss_ax, scatter_ax) = plt.subplots(1, 2, figsize=(10, 4))

    loss_ax.plot(result.g_loss_list, label="Generator")
    loss_ax.plot(result.d_loss_list, label="Discriminator")
    loss_ax.set_title("GAN losses per epoch")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()

    scatter_ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.4, label="Real")
    scatter_ax.scatter(fake_samples[:, 0], fake_samples[:, 1], s=8, alpha=0.4, label="Generated")
    scatter_ax.set_title("Real vs. generated samples")
    scatter_ax.legend()

    fig.tight_layout()
    plot_path = result.model_save_path / "gan_samples.png"
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
