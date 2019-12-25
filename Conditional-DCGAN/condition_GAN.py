import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

MANUAL_SEED = 999
torch.manual_seed(MANUAL_SEED)


class Generator(nn.Module):
    '''
    条件生成器:将条件向量和噪声向量结合,生成跨域向量,从而产生特定条件的样本
    '''

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc_01 = nn.Linear(10, 1000)
        self.fc_02 = nn.Linear(1000 + z_dim, 64 * 28 * 28)
        self.bn_01 = nn.BatchNorm2d(64)
        self.deconv01 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn_02 = nn.BatchNorm2d(32)
        self.deconv02 = nn.ConvTranspose2d(32, 1, 5, 1, 2)
        self.relu = nn.ReLU()

    def forward(self, x, labels):
        batch_size = x.size(0)
        # one-hot标签向量1000维生成条件向量,与噪声向量融合,生成跨域向量
        y_ = self.fc_01(labels)
        y_ = self.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc_02(x)
        # 转换成生成图像的大小
        x = x.view(batch_size, 64, 28, 28)
        x = self.bn_01(x)
        x = self.relu(x)
        x = self.deconv01(x)
        x = self.bn_02(x)
        x = self.relu(x)
        x = self.deconv02(x)
        x = F.sigmoid(x)
        # 生成的图像大小:1,28,28
        return x


class Discriminator(nn.Module):
    '''
    条件判别器,输入包括:输入的样本和对应的条件向量
    '''

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(10, 1000)
        self.fc2 = nn.Linear(64 * 28 * 28 + 1000, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x, labels):
        batch_size = x.size(0)
        # x = x.view(batch_size, 1, 28, 28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(batch_size, 64 * 28 * 28)
        y_ = self.fc1(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


if __name__ == "__main__":
    parse = argparse.ArgumentParser('CGAN')
    parse.add_argument('--batch_size', type=int, default=128)
    parse.add_argument('--lr', type=float, default=0.01)
    parse.add_argument('--epochs', type=int, default=10)
    parse.add_argument('--nz', type=int, default=100)
    parse.add_argument('--save_epoch', type=int, default=1)
    parse.add_argument('--print_iter', type=int, default=50)
    parse.add_argument('--save_dir', type=str, default='/home/liupeng/MrLiu/GAN/CGAN/models')
    parse.add_argument('--samples_dir', type=str, default='/home/liupeng/MrLiu/GAN/CGAN/samples')
    arg = parse.parse_args()

    INPUT_SIZE = 28 * 28
    NUM_LABELS = 10
    device = torch.device('cuda:1')
    real_label = 1
    fake_label = 0
    SAMPLE_NUM = 64

    if not os.path.exists(arg.save_dir):
        os.mkdir(arg.save_dir)
    if not os.path.exists(arg.samples_dir):
        os.mkdir(arg.samples_dir)

    train_dataset = datasets.MNIST(root='../data', train=True, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=arg.batch_size, num_workers=2)

    generator = Generator(arg.nz).to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()

    optim_generator = optim.SGD(params=generator.parameters(), lr=arg.lr)
    optim_discriminator = optim.SGD(params=discriminator.parameters(), lr=arg.lr)

    fixed_noise = torch.randn([SAMPLE_NUM, arg.nz]).to(device)
    fixed_labels = torch.FloatTensor(SAMPLE_NUM, NUM_LABELS).zero_()
    fixed_index = torch.randint(10, [SAMPLE_NUM, 1]).long()
    fixed_labels = fixed_labels.scatter_(1, fixed_index, 1).to(device)

    print('Start Training')
    G_Loss = []
    D_Loss = []
    iter = 0

    for epoch in range(arg.epochs):
        generator.train()
        discriminator.train()

        for batch_idx, (img, label) in enumerate(train_loader):
            img = img.to(device)
            batch_size = img.size(0)
            label = label.view([batch_size, 1]).to(device)
            one_hot_labels = torch.FloatTensor(batch_size, NUM_LABELS).zero_().to(device)
            D_label = torch.ones([batch_size, 1]).to(device)

            '''
            train discriminator
            '''

            # real loss
            optim_discriminator.zero_grad()
            one_hot_labels = one_hot_labels.scatter_(1, label, 1)
            output = discriminator(img, one_hot_labels)
            errD_real = criterion(output, D_label)
            errD_real.backward()
            realD_mean = output.data.cpu().mean()

            # fake loss
            one_hot_labels.fill_(0)

            rand_y = torch.randint(10, [batch_size, 1]).long().to(device)
            one_hot_labels = one_hot_labels.scatter_(1, rand_y, 1)
            noise = torch.randn([batch_size, arg.nz]).to(device)
            D_label.fill_(0)
            g_out = generator(noise, one_hot_labels)
            output = discriminator(g_out.detach(), one_hot_labels)
            errD_fake = criterion(output, D_label)
            fakeD_mean = output.data.cpu().mean()
            errD = errD_real + errD_fake
            errD_fake.backward()
            optim_discriminator.step()

            '''
            train generator
            '''
            optim_generator.zero_grad()
            D_label.fill_(1)
            output = discriminator(g_out, one_hot_labels)
            fakeG_mean = output.data.cpu().mean()
            errG = criterion(output, D_label)
            errG.backward()
            optim_generator.step()

            iter += 1
            G_Loss.append(errG.item())
            D_Loss.append(errD.item())

            if batch_idx % arg.print_iter == 0:
                print('\t {} ({}/{}) D(fake) = {:.4f}, D(real)={:.4f}, G(fake)={:.4f}'.format(epoch, batch_idx, len(train_loader), fakeD_mean, realD_mean, fakeG_mean))

                g_out = generator(fixed_noise, fixed_labels).detach().cpu()
                torchvision.utils.save_image(g_out, '{}/{}_{}.png'.format(arg.samples_dir, epoch, batch_idx))

    plt.figure(figsize=(10, 5))
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.title('G and D Loss')
    plt.plot(G_Loss, label='G')
    plt.plot(D_Loss, label='D')
    plt.legend()
    plt.savefig('Loss.png')