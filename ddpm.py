import os
import torch
from glob import glob
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class SimpleDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform or transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        )
        self.transform = None

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image)
        image = image.convert("RGB")
        image = image.resize((128, 128))

        image = transforms.ToTensor()(image)
        if self.transform:
            image = self.transform(image)
        image = 2 * image - 1
        return image

    def __len__(self):
        return len(self.images)


class SimpleDiffusionModel:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Initialize the diffusion process parameters

        Args:
        - timesteps: Number of diffusion steps
        - beta_start: Starting noise schedule value
        - beta_end: Ending noise schedule value
        """
        self.timesteps = timesteps

        # Create noise schedule (linear beta schedule)
        self.betas = torch.linspace(beta_start, beta_end, timesteps).detach()

        # Compute alphas and cumulative products - detach all fixed parameters
        self.alphas = (1 - self.betas).detach()
        self.alpha_prod = torch.cumprod(self.alphas, 0).detach()
        self.alpha_prod_prev = torch.cat(
            [torch.tensor([1.0]), self.alpha_prod[:-1]]
        ).detach()

        # Calculate quantities for reverse process
        self.sqrt_alpha_prod = torch.sqrt(self.alpha_prod).detach()
        self.sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.alpha_prod).detach()

        # Move all tensors to the correct device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_prod = self.alpha_prod.to(device)
        self.alpha_prod_prev = self.alpha_prod_prev.to(device)
        self.sqrt_alpha_prod = self.sqrt_alpha_prod.to(device)
        self.sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alpha_prod.to(device)

    def forward_diffusion(self, x0, t, noise=None):
        """
        Add noise to the original image at a specific timestep

        Args:
        - x0: Original clean image
        - t: Timestep to add noise
        - noise: Optional predefined noise tensor

        Returns:
        - Noisy image
        - Noise added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Compute noisy image
        sqrt_alpha_prod_t = self.sqrt_alpha_prod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod_t = self.sqrt_one_minus_alpha_prod[t].reshape(
            -1, 1, 1, 1
        )

        noisy_image = sqrt_alpha_prod_t * x0 + sqrt_one_minus_alpha_prod_t * noise

        return noisy_image.to(device), noise.to(device)

    def create_simple_unet(self, in_channels=3, out_channels=3, timesteps=1000):
        """
        Create a simple U-Net for noise prediction

        Args:
        - in_channels: Number of input channels (default 3 for RGB)
        - out_channels: Number of output channels (default 3 for RGB)

        Returns:
        - Simple U-Net model
        """

        class SimpleUNet(nn.Module):
            def __init__(self, in_channels, out_channels, timesteps):
                super().__init__()

                # Time embedding
                self.time_dim = 128 * 128

                def getPositionEncoding(seq_len, d, n=10000):
                    P = np.zeros((seq_len, d))
                    for k in range(seq_len):
                        for i in np.arange(int(d / 2)):
                            denominator = np.power(n, 2 * i / d)
                            P[k, 2 * i] = np.sin(k / denominator)
                            P[k, 2 * i + 1] = np.cos(k / denominator)
                    return P

                # Pre-compute the timestep embeddings
                self.register_buffer(
                    "timestep_embeddings",
                    torch.tensor(
                        getPositionEncoding(timesteps, self.time_dim),
                        dtype=torch.float32,
                    ),
                )

                # Encoder
                self.enc1 = nn.Sequential(
                    nn.Conv2d(in_channels + 1, 64, 3, padding=1),
                    nn.GroupNorm(8, 64),  # Changed from BatchNorm
                    nn.GELU(),  # Changed from ReLU
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.GELU(),
                )

                self.enc2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.GroupNorm(8, 128),
                    nn.GELU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.GroupNorm(8, 128),
                    nn.GELU(),
                )

                self.enc3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.GroupNorm(8, 256),
                    nn.GELU(),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.GroupNorm(8, 256),
                    nn.GELU(),
                )

                # Bridge
                self.bridge = nn.Sequential(
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.GroupNorm(8, 512),
                    nn.GELU(),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.GroupNorm(8, 512),
                    nn.GELU(),
                )

                # Decoder
                self.dec3 = nn.Sequential(
                    nn.Conv2d(512 + 256, 256, 3, padding=1),
                    nn.GroupNorm(8, 256),
                    nn.GELU(),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.GroupNorm(8, 256),
                    nn.GELU(),
                )

                self.dec2 = nn.Sequential(
                    nn.Conv2d(256 + 128, 128, 3, padding=1),
                    nn.GroupNorm(8, 128),
                    nn.GELU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.GroupNorm(8, 128),
                    nn.GELU(),
                )

                self.dec1 = nn.Sequential(
                    nn.Conv2d(128 + 64, 64, 3, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.GELU(),
                    nn.Conv2d(64, out_channels, 3, padding=1),
                )

                self.pool = nn.MaxPool2d(2)
                self.upsample = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )

            def forward(self, x, t):
                # Time embedding
                # x shape : B,3,128,128
                t_emb = self.timestep_embeddings[t]  # 1x128x128
                x = torch.cat([x, t_emb.view(-1, 1, 128, 128)], dim=1)

                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))

                # Bridge
                b = self.bridge(self.pool(e3))

                # Decoder with skip connections
                d3 = self.dec3(torch.cat([self.upsample(b), e3], dim=1))
                d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))

                return d1

        return SimpleUNet(in_channels, out_channels, timesteps).to(device)

    def sample(self, model, shape, device="cpu"):
        """
        Sample from the diffusion model
        Args:
        - model: Trained noise prediction model
        - shape: Shape of the image to generate
        - device: Computation device

        Returns:
        - Generated image
        """
        model.eval()
        img = torch.randn(shape).to(device)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t] * shape[0]).to(device)

            with torch.no_grad():
                noise_pred = model(img, t_tensor)

            # Compute coefficients
            beta_t = self.betas[t]
            sqrt_one_minus_alpha_prod_t = self.sqrt_one_minus_alpha_prod[t]
            sqrt_alpha_t = torch.sqrt(self.alphas[t])

            # Compute mean
            mean = (1 / sqrt_alpha_t) * (
                img - (beta_t / sqrt_one_minus_alpha_prod_t) * noise_pred
            )

            # Add noise for t > 0
            if t > 0:
                noise = torch.randn_like(img)
                sigma = torch.sqrt(beta_t)
                img = mean + sigma * noise
            else:
                img = mean

        return img.clamp(-1, 1)

    def train(self, model, dataloader, epochs=100, lr=1e-3):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(dataloader),
            pct_start=0.1,
        )
        criterion = nn.MSELoss()
        scaler = GradScaler()

        best_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            total_loss = []

            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch in pbar:
                    optimizer.zero_grad()
                    batch = batch.to(device)

                    # Sample random timesteps
                    t = torch.randint(0, self.timesteps, (batch.shape[0],)).to(device)

                    # Add noise to the batch
                    noisy_batch, noise = self.forward_diffusion(batch, t)

                    # Use mixed precision training
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        noise_pred = model(noisy_batch, t)
                        loss = criterion(noise_pred, noise)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    scheduler.step()

                    total_loss.append(loss.item())
                    pbar.set_postfix(
                        {
                            f"Epoch {epoch+1}/{epochs}, Average Loss: {np.mean(total_loss):.6f}"
                            " lr": scheduler.get_last_lr()[0],
                        }
                    )

            avg_loss = np.mean(total_loss)
            # Save best model
            if avg_loss < best_loss:
                print(
                    f"Saving best model with loss: {avg_loss:.6f}, best loss before: {best_loss:.6f}"
                )
                best_loss = avg_loss
                self.save(model, "best_model.pth")

            # Generate samples every 5 epochs
            if (epoch + 1) % 5 == 0:
                os.makedirs("samples", exist_ok=True)
                model.eval()
                with torch.no_grad():
                    samples = self.sample(model, (4, 3, 128, 128), device=device)
                    # Create a grid of images
                    grid = torchvision.utils.make_grid(samples, nrow=2, normalize=True)
                    torchvision.utils.save_image(grid, f"samples/epoch_{epoch+1}.png")

    def save(self, model, path):
        """
        Save the trained model

        Args:
        - model: The trained UNet model
        - path: Path where to save the model
        """
        torch.save(model.state_dict(), path)

    def load(self, model, path):
        """
        Load a trained model

        Args:
        - model: The UNet model architecture
        - path: Path to the saved model weights
        """
        model.load_state_dict(torch.load(path))
        return model


# Example usage placeholder
def main():
    train_images = glob("train/*.jpg")
    train_dataset = SimpleDataset(train_images)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    test_images = glob("test/*.jpg")
    test_dataset = SimpleDataset(test_images)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    # This will be filled in with actual data loading and training code
    timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02

    model = SimpleDiffusionModel(timesteps, beta_start, beta_end)
    unet = model.create_simple_unet(
        in_channels=3, out_channels=3, timesteps=timesteps
    )  # Specify channels explicitly
    print(f"Model Parameters: {sum(p.numel() for p in unet.parameters())}")
    try:
        model.load(unet, "model.pth")
    except:
        print("Training model from scratch")
    model.train(unet, train_loader, epochs=1, lr=1e-4)
    model.save(unet, "model.pth")

    sample = model.sample(unet, (1, 3, 128, 128), device=device)
    plt.imshow(sample[0].detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
