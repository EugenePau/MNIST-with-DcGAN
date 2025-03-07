from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image
import os
import re
import sys
import cv2
import random
import torch
import torchvision
import matplotlib
from torchsummary import summary
from torchvision.transforms import v2
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dset
from tkinter import filedialog


# ignore warnings from PyTorch 
torchvision.disable_beta_transforms_warning()

if __name__ == '__main__':

    class Global:
        folderpath = None 
        image_files = []
        aug_image_files = []
        image_inf = None
        qt_img_label = None
        x_train = None
        y_train = None
        x_test = None
        y_test = None
        classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        label = None
        
        
        CUDA = True
        DATA_PATH = './data'
        BATCH_SIZE = 128
        IMAGE_CHANNEL = 3
        Z_DIM = 100
        G_HIDDEN = 64
        X_DIM = 64
        D_HIDDEN = 64
        EPOCH_NUM = 5
        REAL_LABEL = 1
        FAKE_LABEL = 0
        lr = 2e-4
        seed = 1


    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                # input layer
                nn.ConvTranspose2d(g.Z_DIM, g.G_HIDDEN * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(g.G_HIDDEN * 8),
                nn.ReLU(True),
                # 1st hidden layer
                nn.ConvTranspose2d(g.G_HIDDEN * 8, g.G_HIDDEN * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(g.G_HIDDEN * 4),
                nn.ReLU(True),
                # 2nd hidden layer
                nn.ConvTranspose2d(g.G_HIDDEN * 4, g.G_HIDDEN * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(g.G_HIDDEN * 2),
                nn.ReLU(True),
                # 3rd hidden layer
                nn.ConvTranspose2d(g.G_HIDDEN * 2, g.G_HIDDEN, 4, 2, 1, bias=False),
                nn.BatchNorm2d(g.G_HIDDEN),
                nn.ReLU(True),
                # output layer
                nn.ConvTranspose2d(g.G_HIDDEN, g.IMAGE_CHANNEL, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            return self.main(input)


    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                # 1st layer
                nn.Conv2d(g.IMAGE_CHANNEL, g.D_HIDDEN, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 2nd layer
                nn.Conv2d(g.D_HIDDEN, g.D_HIDDEN * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(g.D_HIDDEN * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 3rd layer
                nn.Conv2d(g.D_HIDDEN * 2, g.D_HIDDEN * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(g.D_HIDDEN * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 4th layer
                nn.Conv2d(g.D_HIDDEN * 4, g.D_HIDDEN * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(g.D_HIDDEN * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # output layer
                nn.Conv2d(g.D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input).view(-1, 1).squeeze(1)



    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # set up CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crete Global variables
    g = Global()

    # Create the generator
    netG = Generator().to(device)
    netG.apply(weights_init)

    # Create the discriminator
    netD = Discriminator().to(device)
    netD.apply(weights_init)


    class VGG19:
        def __init__(self):
            pass
        
        def create_image_grid(self,images, grid_size=(8, 8), image_size=(28, 28)):
            rows, cols = grid_size
            img_height, img_width = image_size

            # Create a blank canvas for the grid
            grid_image = np.zeros((rows * img_height, cols * img_width), dtype=np.uint8)

            # Fill the canvas with images
            for idx, image in enumerate(images):
                row = idx // cols
                col = idx % cols
                grid_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = image

            return grid_image
        
        def converttoOpenCV(self,img):

            # Clip values to the range [0, 1] (for valid visual representation)
            img_np = np.clip(img, 0, 1)
            
            # Resize to 28x28 pixels
            img_resized = cv2.resize(img_np, (28, 28))

            # Convert to uint8 for grid creation
            img_uint8 = (img_resized * 255).astype(np.uint8)
            
            # Convert RGB Images to Grayscale
            img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
            
            return img_gray


        def loadMultiImage(self):
            folderpath = QtWidgets.QFileDialog.getExistingDirectory()
            print(folderpath)
            if folderpath:
                g.folderpath = folderpath
                
                ## File consisting of non-transformed images for tensor conversion in Augmentation step
                raw_image_file = []
                
                # Randomly select 64 images
                image = os.listdir(folderpath)
                image_64 = random.sample(image, 64)
            
                for img_title in image_64:
                    img = cv2.imread(os.path.join(folderpath, img_title))
                    raw_image_file.append(img)
                    
                    # Check if the image is grayscale (2D)
                    if len(img.shape) == 2:  # Grayscale image
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
                    else:  # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                    # OpenCV loads images in the shape (H, W, C), so no need for permute
                    img_np = img / 255.0  # Normalize pixel values to [0, 1]
                    
                    # Convert image into the Numpy format
                    img_np = self.converttoOpenCV(img_np)
                    # Append the Numpy format image into the global set
                    g.image_files.append(img_np)
                    
                return raw_image_file
            return


            
        def showDataAugmentation(self):
            raw_image_files = self.loadMultiImage()

            if raw_image_files is None:
                print("Please load image first.")
                return


            for img in raw_image_files:
                # Converts the OpenCV image to a PyTorch tensor [3, H, W]
                img_tensor = v2.ToImageTensor()(img)  # Use ToImageTensor instead of deprecated ToTensor

                # Transformations
                transforms = v2.Compose([
                    v2.RandomRotation(degrees=40),                         # Randomly rotate ±25 degrees
                    v2.ConvertImageDtype(torch.float32),                   # Convert dtype to float32
                    v2.Normalize(mean=[0.5], std=[0.5])                    # Normalize grayscale images (mean=0.5, std=0.5)
                ])
                img = transforms(img_tensor)  # Apply transformations

                # Denormalize the image for visualization
                def denormalize(img_tensor, mean, std):
                    mean = torch.tensor(mean).view(3, 1, 1)
                    std = torch.tensor(std).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean  # Reverse normalization
                    return img_tensor

                # Denormalize the transformed image
                img_denormalized = denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                # Convert to NumPy and transpose dimensions for Matplotlib (H, W, C)
                img_np = img_denormalized.permute(1, 2, 0).numpy()

                # Convert image into the openCV format
                img_np = self.converttoOpenCV(img_np)

                # Append the augmented image into the global set
                g.aug_image_files.append(img_np)

            # Create the 8x8 grid image
            grid_image = self.create_image_grid(g.image_files, grid_size=(8, 8), image_size=(28, 28))
            grid_image_aug = self.create_image_grid(g.aug_image_files, grid_size=(8, 8), image_size=(28, 28))

            # Display the grid using Matplotlib
            plt.figure(figsize=(8, 8))
            plt.subplot(1,2,1)
            plt.imshow(grid_image, cmap='gray')  # Use cmap='gray' for grayscale images
            plt.axis('off')
            plt.title("Training Dataset (Original)")
            plt.subplot(1,2,2)
            plt.imshow(grid_image_aug, cmap='gray')  # Use cmap='gray' for grayscale images
            plt.axis('off')
            plt.title("Training Dataset (Augmented)")
            plt.show()
            return


        def showModelStructure(self):
            # Ensure the summary call uses the same device
            print(netG)
            print(netD)
            
        def showAccAndLoss(self):
            script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
            image_path = os.path.join(script_dir, 'epoch_25.png')
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display the image using Matplotlib
            plt.figure(figsize=(15, 10))  # Set the figure size
            plt.imshow(img)
            plt.axis('off')  # Turn off axis
            plt.title("Accuracy and Loss Visualization")  # Add a title
            plt.show()

        def inference(self):
            # Ensure Matplotlib is using the correct backend
            matplotlib.use('Qt5Agg')  # Use a backend compatible with PyQt5

            # Data path
            g.DATA_PATH = filedialog.askdirectory(title="Select Root Folder")
            
            # Data preprocessing
            dataset = dset.ImageFolder(
                root=g.DATA_PATH,
                transform=transforms.Compose([
                    transforms.Resize(g.X_DIM),
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images (mean=0.5, std=0.5)
                ])
            )

            # Dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=g.BATCH_SIZE, shuffle=True, num_workers=0
            )

            # Path containing trained model
            script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
            G_model_path = os.path.join(script_dir, 'G_model_epoch_25.pth')
            
            # Load the pre-trained model
            netG.load_state_dict(torch.load(G_model_path, map_location=device))
            
            # Set to eval mode
            netG.eval()

            # Generate noise and fake images
            batch_size = 64
            noise = torch.randn(batch_size, g.Z_DIM, 1, 1, device=device)
            fake_images = netG(noise)
            
            # Grab a batch of real images from the dataloader
            real_batch = next(iter(dataloader))

            # Plot the real images
            plt.figure(figsize=(15, 15))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(
                np.transpose(
                    torchvision.utils.make_grid(
                        real_batch[0].to(device)[:64], padding=5, normalize=True
                    ).cpu(),
                    (1, 2, 0),
                )
            )

            # Plot the fake images
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(
                np.transpose(
                    torchvision.utils.make_grid(
                        fake_images[:64].detach().cpu(), padding=5, normalize=True
                    ),
                    (1, 2, 0),
                )
            )

            # Ensure the plot displays
            plt.show(block=True)

            return
                






    class Window:

        def __init__(self):
            self.windowHeight = 720
            self.UnitWIDTH = 250
            self.UnitWIDTHWithSpace = self.UnitWIDTH+10
            self.vgg19 = VGG19()

            self.app = QtWidgets.QApplication(sys.argv)
            self.window = QtWidgets.QWidget()
            self.window.setWindowTitle('2024 CvDl Hw2 Q2')
            self.window.resize(self.UnitWIDTHWithSpace*3, self.windowHeight)
            self.showImg = None
            self.boxA()


        def boxA(self):
            btn1 = QtWidgets.QPushButton(self.window)
            btn1.setText('1. Show Data Augmentation')
            btn1.clicked.connect(self.vgg19.showDataAugmentation)
            
            btn2 = QtWidgets.QPushButton(self.window)
            btn2.setText('2. Show Model Structure')
            btn2.clicked.connect(self.vgg19.showModelStructure)

            btn3 = QtWidgets.QPushButton(self.window)
            btn3.setText('3. Show Accuracy and Loss')
            btn3.clicked.connect(self.vgg19.showAccAndLoss)

            btn4 = QtWidgets.QPushButton(self.window)
            btn4.setText('4. Inference')
            btn4.clicked.connect(self.vgg19.inference)

            box = QtWidgets.QGroupBox(title="5. VGG19 Test", parent=self.window)
            box.setGeometry(0, 0, self.UnitWIDTH, self.windowHeight)
            layout = QtWidgets.QVBoxLayout(box)
            layout.addWidget(btn1)
            layout.addWidget(btn2)
            layout.addWidget(btn3)
            layout.addWidget(btn4)


        def render(self):
            self.window.show()
            sys.exit(self.app.exec())


    window = Window()
    window.render()
