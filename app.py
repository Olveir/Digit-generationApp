import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# One-hot encoding
def one_hot(label, num_classes=10):
    return torch.eye(num_classes)[label]

# CVAE Model
class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + 10, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid()
        )

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        return self.decoder(zy).view(-1, 1, 28, 28)

@st.cache_resource
def load_model():
    model = CVAE()
    model.load_state_dict(torch.load("models/cvae_mnist.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

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
