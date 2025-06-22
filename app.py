import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load the trained model
class DigitGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64 + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = self.encoder(x.view(-1, 28 * 28))
        x = torch.cat([x, labels], dim=1)
        x = self.decoder(x)
        return x.view(-1, 1, 28, 28)

@st.cache_resource
def load_model():
    model = DigitGenerator()
    model.load_state_dict(torch.load("models/mnist_generator.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def one_hot(label, num_classes=10):
    return torch.eye(num_classes)[label]

st.title("MNIST Digit Generator")
digit = st.number_input("Enter a digit (0–9)", min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    model = load_model()
    imgs = []
    for _ in range(5):
        noise = torch.rand(1, 1, 28, 28)
        label = one_hot(torch.tensor([digit]))
        gen = model(noise, label).detach().numpy().squeeze()
        imgs.append(gen)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axs):
        ax.imshow(imgs[i], cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
