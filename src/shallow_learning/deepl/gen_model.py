# deepl/gen_model.py
# deepl/gen_model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# VAE
# ─────────────────────────────────────────────────────────────────────────────

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 8  -> 4
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4  -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8  -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 32 -> 64
            nn.Tanh()                                # output in [-1, 1]
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.net(h)


class VAE(nn.Module):
    """
    Variational AutoEncoder for 64x64 RGB images.
    Encodes input to (mu, logvar), samples latent z, decodes back to image.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def sample(self, n, device):
        """Generate n random images from prior."""
        z = torch.randn(n, self.latent_dim).to(device)
        with torch.no_grad():
            return self.decoder(z)


# ─────────────────────────────────────────────────────────────────────────────
# GAN
# ─────────────────────────────────────────────────────────────────────────────

class GANGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4  -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8  -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 32 -> 64
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.net(h)


class GANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    # 64 -> 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),   # 32 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), # 8  -> 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class GAN(nn.Module):
    """
    Generative Adversarial Network for 64x64 RGB images.
    Contains Generator and Discriminator as submodules.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim   = latent_dim
        self.generator    = GANGenerator(latent_dim)
        self.discriminator = GANDiscriminator()

    def forward(self, z):
        return self.generator(z)

    def sample(self, n, device):
        """Generate n random images from noise."""
        z = torch.randn(n, self.latent_dim).to(device)
        with torch.no_grad():
            return self.generator(z)


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion Model
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionUNet(nn.Module):
    """
    Simple UNet-style noise predictor for DDPM.
    Predicts the noise added at timestep t.
    """
    def __init__(self, time_emb_dim=64):
        super().__init__()

        # Time embedding: scalar t -> vector
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder — all same-size convs, downsampling via MaxPool
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU())    # 64x64
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU())   # 32x32
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU())  # 16x16
        self.enc4 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU()) # 8x8

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck time injection
        self.time_proj = nn.Linear(time_emb_dim, 256)

        # Decoder with skip connections
        self.up3    = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3   = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, 1, 1), nn.ReLU()) # 16x16

        self.up2    = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2   = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, 1, 1), nn.ReLU())   # 32x32

        self.up1    = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1   = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, 1, 1), nn.ReLU())    # 64x64

        self.out    = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.float().unsqueeze(1) / 1000.0)

        # Encoder
        e1 = self.enc1(x)                # 64x64
        e2 = self.enc2(self.pool(e1))    # 32x32
        e3 = self.enc3(self.pool(e2))    # 16x16
        e4 = self.enc4(self.pool(e3))    # 8x8

        # Inject time embedding into bottleneck
        t_proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        e4 = e4 + t_proj                 # 8x8

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))  # 16x16
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # 32x32
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # 64x64

        return self.out(d1)

class DiffusionModel(nn.Module):
    """
    DDPM-style Diffusion Model for 64x64 RGB images.
    Uses a UNet to predict noise at each timestep.
    """
    def __init__(self, T=1000, time_emb_dim=64):
        super().__init__()
        self.T      = T
        self.unet   = DiffusionUNet(time_emb_dim)

        # Pre-compute noise schedule (linear beta schedule)
        betas          = torch.linspace(1e-4, 0.02, T)
        alphas         = 1.0 - betas
        alpha_bar      = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas',     betas)
        self.register_buffer('alphas',    alphas)
        self.register_buffer('alpha_bar', alpha_bar)

    def forward(self, x0, t):
        """Add noise to x0 at timestep t and return noisy image + noise."""
        noise     = torch.randn_like(x0)
        ab        = self.alpha_bar[t].view(-1, 1, 1, 1)
        x_t       = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
        pred_noise = self.unet(x_t, t)
        return pred_noise, noise

    @torch.no_grad()
    def sample(self, n, device):
        """Generate n images via reverse diffusion (DDPM sampling)."""
        x = torch.randn(n, 3, 64, 64).to(device)
        for t in reversed(range(self.T)):
            t_batch   = torch.full((n,), t, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t_batch)
            alpha      = self.alphas[t]
            alpha_bar  = self.alpha_bar[t]
            beta       = self.betas[t]

            # DDPM reverse step
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
            )
            if t > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)

        return x.clamp(-1, 1)

# ─────────────────────────────────────────────────────────────────────────────
# GenModelTrainer
# ─────────────────────────────────────────────────────────────────────────────


class GenModelTrainer:
    """
    Model-agnostic trainer for VAE, GAN, and DiffusionModel.
    Handles different training logic per model type internally.
    """
    def __init__(self, model, device, lr=2e-4, save_dir="scripts/output_genmodel"):
        self.model    = model.to(device)
        self.device   = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.model_type = self._detect_model_type()
        print(f"Trainer initialized for: {self.model_type}")

        # ── Optimizers ────────────────────────────────────────────────────────
        if self.model_type == "GAN":
            self.opt_g = optim.Adam(model.generator.parameters(),     lr=lr, betas=(0.5, 0.999))
            self.opt_d = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        elif self.model_type == "VAE":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        elif self.model_type == "Diffusion":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.loss_history = []

    def _detect_model_type(self):
        if isinstance(self.model, VAE):
            return "VAE"
        elif isinstance(self.model, GAN):
            return "GAN"
        elif isinstance(self.model, DiffusionModel):
            return "Diffusion"
        else:
            raise ValueError("Unknown model type. Must be VAE, GAN, or DiffusionModel.")

    # ── VAE loss ──────────────────────────────────────────────────────────────
    def _vae_loss(self, recon, x, mu, logvar):
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kld        = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kld) / x.size(0)

    # ── Train one epoch ───────────────────────────────────────────────────────
    def _train_epoch_vae(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            x = batch.to(self.device)
            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(x)
            loss = self._vae_loss(recon, x, mu, logvar)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def _train_epoch_gan(self, dataloader):
        self.model.train()
        total_g_loss = 0.0
        total_d_loss = 0.0
        criterion    = nn.BCELoss()

        for batch in dataloader:
            real = batch.to(self.device)
            b    = real.size(0)

            real_labels = torch.ones(b, 1).to(self.device)
            fake_labels = torch.zeros(b, 1).to(self.device)

            # ── Train Discriminator ───────────────────────────────────────────
            self.opt_d.zero_grad()
            d_real = self.model.discriminator(real)
            d_real_loss = criterion(d_real, real_labels)

            z    = torch.randn(b, self.model.latent_dim).to(self.device)
            fake = self.model.generator(z).detach()
            d_fake = self.model.discriminator(fake)
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            self.opt_d.step()

            # ── Train Generator ───────────────────────────────────────────────
            self.opt_g.zero_grad()
            z    = torch.randn(b, self.model.latent_dim).to(self.device)
            fake = self.model.generator(z)
            g_out  = self.model.discriminator(fake)
            g_loss = criterion(g_out, real_labels)
            g_loss.backward()
            self.opt_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        n = len(dataloader)
        return (total_g_loss + total_d_loss) / (2 * n)

    def _train_epoch_diffusion(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            x = batch.to(self.device)
            t = torch.randint(0, self.model.T, (x.size(0),), device=self.device)
            self.optimizer.zero_grad()
            pred_noise, noise = self.model(x, t)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    # ── Save ONNX ─────────────────────────────────────────────────────────────
    def save_onnx(self, filename=None):
        self.model.eval()
        if filename is None:
            filename = os.path.join(self.save_dir, f"{self.model_type.lower()}_model.onnx")

        if self.model_type == "VAE":
            dummy = torch.randn(1, 3, 64, 64).to(self.device)
            torch.onnx.export(self.model.decoder,
                              torch.randn(1, self.model.latent_dim).to(self.device),
                              filename,
                              input_names=['z'], output_names=['image'],
                              dynamic_axes={'z': {0: 'batch'}, 'image': {0: 'batch'}})

        elif self.model_type == "GAN":
            dummy = torch.randn(1, self.model.latent_dim).to(self.device)
            torch.onnx.export(self.model.generator, dummy, filename,
                              input_names=['z'], output_names=['image'],
                              dynamic_axes={'z': {0: 'batch'}, 'image': {0: 'batch'}})

        elif self.model_type == "Diffusion":
            dummy_x = torch.randn(1, 3, 64, 64).to(self.device)
            dummy_t = torch.zeros(1, dtype=torch.long).to(self.device)
            torch.onnx.export(self.model.unet, (dummy_x, dummy_t), filename,
                              input_names=['x', 't'], output_names=['noise'],
                              dynamic_axes={'x': {0: 'batch'}, 'noise': {0: 'batch'}})

        print(f"ONNX saved to {filename}")

    def save_pt(self, filename=None):
        if filename is None:
            filename = os.path.join(self.save_dir, f"{self.model_type.lower()}_best.pt")
        torch.save(self.model.state_dict(), filename)
        print(f"PT weights saved to {filename}")

    # ── Save loss plot ────────────────────────────────────────────────────────
    def save_plot(self, filename=None):
        if filename is None:
            filename = os.path.join(self.save_dir, f"{self.model_type.lower()}_loss.png")
        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history, marker='o', markersize=2)
        plt.title(f"{self.model_type} Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved to {filename}")

    # ── Main train loop ───────────────────────────────────────────────────────
    def train(self, dataloader, epochs, save_every=10):
        for epoch in range(1, epochs + 1):

            if self.model_type == "VAE":
                loss = self._train_epoch_vae(dataloader)
            elif self.model_type == "GAN":
                loss = self._train_epoch_gan(dataloader)
            elif self.model_type == "Diffusion":
                loss = self._train_epoch_diffusion(dataloader)

            self.loss_history.append(loss)
            print(f"Epoch [{epoch}/{epochs}]  Loss: {loss:.4f}")

            # Save intermediate ONNX every save_every epochs
            if epoch % save_every == 0:
                onnx_path = os.path.join(
                    self.save_dir,
                    f"{self.model_type.lower()}_epoch{epoch}.onnx"
                )
                self.save_onnx(onnx_path)

        # Final save
        self.save_onnx()
        self.save_pt()
        self.save_plot()
