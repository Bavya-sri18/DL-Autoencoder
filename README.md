# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

In practical scenarios, images often contain noise that degrades the performance of computer vision models. A convolutional autoencoder learns compressed representations of images and reconstructs them, which can be used to remove noise.

Dataset: MNIST (28×28 grayscale images of handwritten digits)
Noise: Gaussian noise will be added to simulate real-world scenarios

## DESIGN STEPS
### STEP 1: 

Import required libraries: PyTorch, torchvision, matplotlib, and others for data handling and visualization.

### STEP 2: 

Download the MNIST dataset and apply transformations to convert images to tensors suitable for training.

### STEP 3: 

Add Gaussian noise to the training and testing images using a custom noise-adding function.

### STEP 4: 

Encoder: Convolutional layers (Conv2D) with ReLU activations and MaxPooling
Decoder: Transposed convolutional layers (ConvTranspose2D) with ReLU and Sigmoid activations to reconstruct the image

### STEP 5: 

Initialize the autoencoder model
Define Mean Squared Error (MSE) as the loss function
Choose Adam optimizer for training

### STEP 6: 

Train the autoencoder using the noisy images as input and the original clean images as the target. Track the loss over epochs to monitor learning.

### STEP 7:

Compare the original, noisy, and denoised images
Visualize results to assess the model’s performance in removing noise


## PROGRAM

### Name: BAVYA SRI B

### Register Number: 212224230034

```python
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x


# Initialize model
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training function
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name: BAVYA SRI B")
    print("Register Number: 212224230034")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)


            #forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            #Backward pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):4f}")
# Visualization function
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: BAVYA SRI B ")
    print("Register Number: 212224230034")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


```

### OUTPUT

### Model Summary


### Training loss

## Original vs Noisy Vs Reconstructed Image



## RESULT
The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
