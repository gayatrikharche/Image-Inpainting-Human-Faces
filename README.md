# Image Inpainting with Deep Learning
This project explores three architectures for image inpainting, a challenging problem in computer vision aimed at reconstructing missing or deteriorated parts of an image. Our method, inspired by Context Encoders, leverages unsupervised visual feature learning by predicting pixel context to reconstruct missing regions.

## Architectures
- Autoencoder1: A basic convolutional autoencoder with an encoder-decoder structure for fundamental inpainting tasks.
- Autoencoder2: An enhanced autoencoder with additional layers and depth for capturing more complex features.
- UNet: A sophisticated architecture with symmetrical encoder-decoder pathways and skip connections for detailed and precise inpainting.
  
## Dataset
Human Faces Dataset: Sourced from Kaggle, featuring diverse human face images under various conditions. Preprocessing included resizing, augmentation, and custom dataset loading using PyTorch.

## Results
- Autoencoder1: Moderate reconstruction loss, capturing basic features but lacking fine details.
- Autoencoder2: Lower reconstruction loss with better detail and structural coherence.
- UNet: Achieved the lowest reconstruction loss, offering the most detailed and visually appealing inpainted images.
