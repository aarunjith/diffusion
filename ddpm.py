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

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class SimpleDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image)
        image = image.convert("RGB")
        image = image.resize((64, 64))
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
        image = image / 127.5 - 1
        if self.transform:
            image = self.transform(image)
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

    def create_simple_unet(self, in_channels=3, out_channels=3):
        """
        Create a simple U-Net for noise prediction

        Args:
        - in_channels: Number of input channels (default 3 for RGB)
        - out_channels: Number of output channels (default 3 for RGB)

        Returns:
        - Simple U-Net model
        """

        class SimpleUNet(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                # Encoder
                self.enc1 = nn.Sequential(
                    nn.Conv2d(in_channels + 1, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                )
                self.pool1 = nn.MaxPool2d(2)

                self.enc2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                )
                self.pool2 = nn.MaxPool2d(2)

                # Bridge
                self.bridge = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256)
                )

                # Decoder
                self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec1 = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),  # 256 because of skip connection
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                )

                self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec2 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),  # 128 because of skip connection
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.Conv2d(64, out_channels, 3, padding=1),
                )

            def forward(self, x, t):
                # Time embedding
                t_emb = torch.sin(t.float() / 100)
                t_emb = t_emb.view(-1, 1, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
                x = torch.cat([x, t_emb], dim=1)

                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool1(e1))

                # Bridge
                b = self.bridge(self.pool2(e2))

                # Decoder with skip connections
                d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))

                return d2

        return SimpleUNet(in_channels, out_channels).to(device)

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

    def train(self, model, dataloader, epochs=100, lr=2e-4):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        for epoch in tqdm(range(epochs)):
            model.train()
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                batch = batch.to(device)

                # Sample random timesteps
                t = torch.randint(0, self.timesteps, (batch.shape[0],)).to(device)

                # Add noise to the batch
                noisy_batch, noise = self.forward_diffusion(batch, t)

                # Predict noise
                noise_pred = model(noisy_batch, t)

                # Compute loss
                loss = criterion(noise_pred, noise)
                loss.backward()

                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            if (epoch) % 10 == 0:
                os.makedirs("samples", exist_ok=True)
                sample = self.sample(model, (1, 3, 64, 64), device=device)
                print(f"Saving sample {epoch+1}")
                plt.imsave(
                    f"samples/sample_{epoch+1}.png",
                    ((sample[0] + 1) * 127.5 / 255)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0),
                )

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
    beta_start = 0.0001
    beta_end = 0.02

    model = SimpleDiffusionModel(timesteps, beta_start, beta_end)
    unet = model.create_simple_unet(
        in_channels=3, out_channels=3
    )  # Specify channels explicitly
    print(f"Model Parameters: {sum(p.numel() for p in unet.parameters())}")
    try:
        model.load(unet, "model.pth")
    except:
        print("Training model from scratch")
    model.train(unet, train_loader, epochs=100, lr=1e-3)
    model.save(unet, "model.pth")

    sample = model.sample(unet, (1, 3, 64, 64), device=device)
    plt.imshow(sample[0].detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
