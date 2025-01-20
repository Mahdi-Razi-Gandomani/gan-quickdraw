# GAN for doodle Generation

This project implements a Generative Adversarial Network (GAN) to generate doodles. The GAN consists of two neural networks: a **generator** that creates images and a **discriminator** that distinguishes between real and generated images. The two networks are trained simultaneously in a competitive manner.

---

## Features

- **Generator**:
  - Takes random noise as input and generates images.
  - Uses transposed convolutional layers to upsample the input noise into a 28x28 grayscale image.

- **Discriminator**:
  - Takes images as input and classifies them as real or fake.
  - Uses convolutional layers to extract features and make predictions.

- **Adversarial Training**:
  - The generator and discriminator are trained simultaneously.
  - The generator learns to create realistic images, while the discriminator learns to distinguish between real and fake images.

- **Image Visualization**:
  - Generated images are visualized during training to monitor progress.
