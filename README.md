# MNIST-with-DcGAN
Training a Deep Convolutional Generative Adversarial Network (DcGAN) to Generate Handwritten Digits

GAN Generated MNIST Digits

------------------------------------------------------------------------------------------------------------------

📌 Overview

This project implements a Deep Convolutional Generative Adversarial Network (DcGAN) to generate handwritten digits similar to those in the MNIST dataset. The model learns the distribution of MNIST digits and generates synthetic samples resembling real digits.

------------------------------------------------------------------------------------------------------------------

⚡ Features

✅ Implements a DcGAN using PyTorch/TensorFlow

✅ Trains a Generator and Discriminator from scratch

✅ Uses the MNIST dataset for training

✅ Generates new synthetic digits after training

✅ Supports visualization of training progress and generated images

------------------------------------------------------------------------------------------------------------------

📂 Project Structure
```
MNIST-with-DcGAN/
│── data/                 # Dataset (if applicable)
│── models/               # Saved trained models
│── output_samples/       # Generated images from the GAN
│── src/                  # Source code
│   ├── main.py           # Main script to train and test the GAN
│   ├── mnist_train.py    # Handles dataset loading and preprocessing
│── requirements.txt      # Dependencies list
│── README.md             # Project documentation
│── LICENSE               # MIT License
```
------------------------------------------------------------------------------------------------------------------

📦 Installation & Setup

🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/EugenePau/MNIST-with-DcGAN.git
cd MNIST-with-DcGAN
```

🔹 Step 2: Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

🔹 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```
------------------------------------------------------------------------------------------------------------------

🚀 Training the Model

Run the following command to train the DcGAN on the MNIST dataset:


```bash
python main.py --epochs 50 --batch_size 128
(Modify epochs and batch_size as needed.)
```

📈 Example Generated Images

🖼️ Sample Output from the Trained GAN

After training, the generator produces images like these:

![Generated Digits](images/generated_digits.png)


------------------------------------------------------------------------------------------------------------------

🤝 Contributing
Want to improve this project? Feel free to fork the repo and submit a pull request!

Fork the repository

Create a feature branch (git checkout -b feature-branch-name)

Commit changes (git commit -m "Added new feature")

Push to GitHub (git push origin feature-branch-name)

Create a Pull Request
