import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Hiperparâmetros de difusão
T = 100
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# One-hot
def one_hot(y, num_classes=10):
    return torch.eye(num_classes)[y]

# Modelo UNet simples condicional
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_label = nn.Linear(10, 28*28)
        self.model = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    
    def forward(self, x, labels):
        label_img = self.fc_label(labels).view(-1, 1, 28, 28)
        x_cat = torch.cat([x, label_img], dim=1)
        return self.model(x_cat)

# Carregar modelo
@st.cache_resource
def load_model():
    model = SimpleUNet()
    model.load_state_dict(torch.load("models/diffusion_mnist.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Função de amostragem
def sample(model, digit, n=5):
    x = torch.randn(n, 1, 28, 28)
    label = one_hot(torch.tensor([digit] * n))

    for t in reversed(range(T)):
        z = torch.randn_like(x) if t > 0 else 0
        beta_t = beta[t]
        alpha_t = alpha[t]
        alpha_hat_t = alpha_hat[t]

        with torch.no_grad():
            eps_theta = model(x, label)

        x = (1 / alpha_t.sqrt()) * (
            x - ((1 - alpha_t) / (1 - alpha_hat_t).sqrt()) * eps_theta
        ) + beta_t.sqrt() * z
    return x
# App
st.title("Handwritten Digit Image Generator")
digit = st.number_input("Enter digit (0–9)", min_value=0, max_value=9)

if st.button("Generate"):
    model = load_model()
    label = one_hot(torch.tensor([digit]*5))
    z = torch.randn(5, model.latent_dim)
    with torch.no_grad():
        samples = model.decode(z, label).squeeze().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axs):
        ax.imshow(samples[i], cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
