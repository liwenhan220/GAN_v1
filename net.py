from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
import numpy as np
import cv2
from torch.nn import BCELoss
from torch.optim import Adam

NOISE_DIM = 100
MINIBATCH_SIZE = 3000
GEN_INPUT_SHAPE = 100
loss_fn = BCELoss()

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, (3,3))
#         self.relu = nn.ReLU()

#         self.conv2 = nn.Conv2d(32, 64, (3,3))

#         self.max_pool = nn.MaxPool2d((3,3))
#         self.flatten = nn.Flatten()
#         self.mlp = nn.Sequential(
#             nn.Linear(8*8*64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)

#         x = self.conv2(x)
#         x = self.relu(x)

#         x = self.max_pool(x)
#         x = self.flatten(x)
#         return self.mlp(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.mlp(self.flatten(x))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(NOISE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x.reshape(x.shape[0], 1, 28, 28)


def load_data():
    training_data = datasets.MNIST(root='mnist', transform=ToTensor(), download=True)
    return training_data

def sample_real_img(dataset):
    imgs = []
    for _ in range(MINIBATCH_SIZE // 2):
        imgs.append(dataset[np.random.randint(0, len(dataset))][0])
    return torch.stack(imgs)

def sample_fake_img(generator):
    noises = torch.normal(0, 1, (MINIBATCH_SIZE // 2, NOISE_DIM))
    return generator(noises)

def show_img(img):
    cv2.imshow('img', cv2.resize(img.reshape((28,28)), (500, 500)))
    cv2.waitKey(1)
    input('PRESS ANY KEY')

def get_acc(y_pred, y):
    num = 0
    for i in range(len(y)):
        if torch.round(y_pred[i][0]) == y[i][0]:
            num += 1
    return num / len(y)


def train_disc(dataset, generator, discriminator, d_opt, episode):
    for e in range(episode):
        generator.eval()
        discriminator.train()
        X = []
        y = []

        X.append(sample_real_img(dataset))
        X.append(sample_fake_img(generator).detach())
        X = torch.cat(X)

        y.append(torch.ones((MINIBATCH_SIZE//2, 1)))
        y.append(torch.zeros((MINIBATCH_SIZE//2, 1)))
        y = torch.cat(y)

        d_opt.zero_grad()

        outputs = discriminator(X)

        loss = loss_fn(outputs, y)
        loss.backward()

        d_opt.step()

        print('Discriminator acc: {}, num episode: {}'.format(get_acc(outputs, y), e))

def train_gen(generator, discriminator, g_opt, episode):
    for e in range(episode):
        generator.train()
        discriminator.eval()
        X = torch.normal(0, 1, (MINIBATCH_SIZE, NOISE_DIM))
        y = torch.ones((MINIBATCH_SIZE, 1))

        g_opt.zero_grad()

        imgs = generator(X)

        outputs = discriminator(imgs)

        loss = loss_fn(outputs, y)

        loss.backward()

        g_opt.step()
        print('GGenerator acc: {}, num episode: {}'.format(get_acc(outputs, y), e))

def save_model(nn, name):
    torch.save(nn.state_dict(), name)

def load_model(nn, name):
    nn.load_state_dict(torch.load(name))

def testSample():
    dataset = load_data()
    gen = Generator()
    disc = Discriminator()
    train_disc(dataset, gen, disc)


def testNN():

    import cv2

    training_data = datasets.MNIST(root='mnist', transform=ToTensor(), download=True)

    img = training_data[1][0]


    sample_img = img.numpy().reshape((28,28))

    noise = torch.normal(0, 1, (2, NOISE_DIM))

    disc = Discriminator()
    gen = Generator()

    print(img.shape)
    print(disc(img))

    gen_img = gen(noise).detach().numpy()

    print(gen_img.shape)
    # cv2.imshow('img', cv2.resize(gen_img.reshape((28,28)), (500, 500)))
    # cv2.waitKey(1)

    # input('PRESS ANY KEY TO END')

# testSample()

if __name__ == '__main__':
    dataset = datasets.MNIST(root='mnist', transform=ToTensor(), download=True)

    disc = Discriminator()
    gen = Generator()

    d_opt = Adam(disc.parameters(), lr=0.0002)
    g_opt = Adam(gen.parameters(), lr=0.0002)

    for _ in range(100000000000):
        train_disc(dataset, gen, disc, d_opt, 10)
        train_gen(gen, disc, g_opt, 10)

        save_model(gen, 'gen')
        save_model(disc, 'disc')


