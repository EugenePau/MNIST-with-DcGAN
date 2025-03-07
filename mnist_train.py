import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tkinter import filedialog

if __name__ == '__main__':
    class Global:
        CUDA = True
        DATA_PATH = None
        BATCH_SIZE = 64
        IMAGE_CHANNEL = 3
        Z_DIM = 100
        G_HIDDEN = 64
        X_DIM = 64
        D_HIDDEN = 64
        EPOCH_NUM = 25
        REAL_LABEL = 0.9
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
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)
            m.bias.data.fill_(0)

    # set up CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crete Global variables
    g = Global()
    
    # data path
    g.DATA_PATH = filedialog.askdirectory(title="Select Root Folder")

    # Create the generator
    netG = Generator().to(device)
    netG.apply(weights_init)

    # Create the discriminator
    netD = Discriminator().to(device)
    netD.apply(weights_init)
    
    # Data Loading
    # Data preprocessing
    dataset = dset.ImageFolder(root=g.DATA_PATH, transform = transforms.Compose([
        transforms.RandomRotation(degrees=25),                         # Randomly rotate ±25 degrees
        transforms.Resize(g.X_DIM),
        transforms.ToTensor(),                                         # Convert PIL image to tensor
        transforms.ConvertImageDtype(torch.float32),                   # Convert dtype to float32
        transforms.Normalize(mean=[0.5], std=[0.5])                    # Normalize grayscale images (mean=0.5, std=0.5)
                            ]))

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=g.BATCH_SIZE,
                                            shuffle=True, num_workers=2)

    ## Set up a loss function and optimizer

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that I will use to visualize the progression of the generator
    viz_noise = torch.randn(g.BATCH_SIZE, g.Z_DIM, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    

    print("Starting Training Loop...")
    for epoch in range(g.EPOCH_NUM):
        for i, data in enumerate(dataloader, 0):

            #### (1) Update the discriminator with real data ####
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            real_cpu = real_cpu + 0.05 * torch.randn_like(real_cpu)  # Add noise to real images
            b_size = real_cpu.size(0)
            # Create real labels
            real_label = torch.full((b_size,), g.REAL_LABEL, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output_real = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output_real, real_label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output_real.mean().item()

            #### (2) Update the discriminator with fake data ####
            # Generate batch of latent vectors
            noise = torch.randn(b_size, g.Z_DIM, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            fake = fake + 0.05 * torch.randn_like(fake)  # Add noise to fake images
            # Create fake labels
            fake_label = torch.full((b_size,), g.FAKE_LABEL, dtype=torch.float, device=device)
            # Classify all fake batch with D
            output_fake = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output_fake, fake_label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            #### (3) Update the generator with fake data ####
            netG.zero_grad()
            # Create real labels for generator loss
            real_label = torch.full((b_size,), g.REAL_LABEL, dtype=torch.float, device=device)  # Fake images should be classified as real
            # Generate new fake images
            fake = netG(noise)
            # Pass fake images through discriminator
            output_fake = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output_fake, real_label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output_fake.mean().item()
            # Update G
            optimizerG.step()

            #### (4) Save Losses for plotting later ####
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Output training stats
            if i % 100 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, g.EPOCH_NUM, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == g.EPOCH_NUM-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(viz_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            # Checkpoint Saving (Model)
            if (epoch + 1) % 5 == 0:
                torch.save(netG.state_dict(), f'./G_model_epoch_{epoch + 1}.pth')
                torch.save(netD.state_dict(), f'./D_model_epoch_{epoch + 1}.pth')
            
            # Checkpoint Saving (Image)
            if (iters+1) % 1000 == 0:
                plt.figure(figsize=(15,15))
                plt.axis("off")
                plt.title(f"Fake Images at {iters+1} iterations")
                plt.imshow(np.transpose(img_list[-1],(1,2,0)))
                plt.savefig(f"./iteration_{iters+1}.png")
                plt.close()  # Close the figure to free memory
            
            
            iters += 1
    
    

    
    # Plot the loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='G') 
    plt.plot(D_losses, label='D') 
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss During Training")
    plt.legend()


    
    # Save the image
    plt.savefig(f"./epoch_{g.EPOCH_NUM}.png")
    # Show the image
    plt.show()

    # Save the model
    torch.save(netD.state_dict(), f"./D_model.pth")
    torch.save(netG.state_dict(), f"./G_model.pth")


    ##########################

