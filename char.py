import gzip
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

class Dataset:

    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    item_count = [60000, 60000, 10000, 10000]
    image_size = 28**2

    def __init__(self, data_location):
        f = list(map(lambda a: gzip.open(a, 'r'), map(lambda a: data_location + '\\' + a, Dataset.files)))
        for i in range(4):
            if i % 2 == 0:
                f[i].read(16)
            else:
                f[i].read(8)
        buf1 = [f[i].read(Dataset.image_size * Dataset.item_count[i]) for i in range(0, 4, 2)]
        buf2 = [f[i].read(Dataset.item_count[i]) for i in range(1, 4, 2)]

        images = [np.frombuffer(buf1[i], dtype=np.uint8).astype(np.float32).reshape(Dataset.item_count[2*i],
                                                                                  Dataset.image_size) for i in range(2)]
        self.train_images = images[0]
        self.test_images = images[1]
        labels = [np.frombuffer(buf2[i], dtype=np.uint8).astype(np.int64) for i in range(2)]
        self.train_labels = labels[0]
        self.test_labels = labels[1]

    def preprocess_images(self):
        train_means = np.mean(self.train_images, axis=1, keepdims=True)
        train_dev = np.std(self.train_images, axis=1, keepdims=True)

        self.train_images = (self.train_images - train_means) / train_dev

        test_means = np.mean(self.test_images, axis=1, keepdims=True)
        test_dev = np.std(self.test_images, axis=1, keepdims=True)

        self.test_images = (self.test_images - test_means) / test_dev


class Network:

    h1 = 300
    h2 = 50

    def __init__(self, data=None):
        if data:
            pass
        else:
            self.M1 = rng.normal(0, (2/784)**0.5, (Network.h1, 784))
            self.b1 = np.zeros(Network.h1)
            self.o1 = np.zeros((Network.h1, Teacher.handful))
            self.M2 = rng.normal(0, (2/Network.h1)**0.5, (Network.h2, Network.h1))
            self.b2 = np.zeros(Network.h2)
            self.o2 = np.zeros((Network.h2, Teacher.handful))
            self.M3 = rng.normal(0, (2/Network.h2)**0.5, (10, Network.h2))
            self.b3 = np.zeros(10)
            self.o3 = np.zeros((10, Teacher.handful))

    def run(self, picture):
        return softmaxfn(self.M3 @ rectifier(self.M2 @ rectifier(self.M1 @ picture + self.b1) + self.b2) + self.b3) 

    def run_handful(self, picture):
        self.o1 = rectifier(self.M1 @ picture + self.b1[:, np.newaxis])
        self.o2 = rectifier(self.M2 @ self.o1 + self.b2[:, np.newaxis])
        self.o3 = softmaxfn(self.M3 @ self.o2 + self.b3[:, np.newaxis])
        return self.o3


def rectifier(vec):
    return np.maximum(vec, 0)

def softmaxfn(vec):
    return np.exp(vec) / np.sum(np.exp(vec), axis=0)


class Teacher:
    
    step = 0.005
    handful = 10

    def __init__(self, model, dataset):
        self.train_images, self.train_labels = dataset.train_images, dataset.train_labels
        self.model = model

        self.errors = []


    def train_epoch(self):
        shuffle = rng.permutation(self.train_images.shape[0])

        for i in range(0, (
                self.train_images.shape[0] - Teacher.handful if self.train_images.shape[0] % Teacher.handful else self.train_images.shape[0]),
                       Teacher.handful):
            self.train_handful(shuffle, i)

        er = np.array(self.errors)
        print(f'average error: {np.mean(er)} worst-case error at: {np.argmax(er)}')
        self.errors = []


    def train_handful(self, shuffle, ind):
        picture = self.train_images[shuffle[ind:(ind + Teacher.handful)]].T
        n = self.train_labels[shuffle[ind:(ind + Teacher.handful)]]
        self.model.run_handful(picture)

        self.errors.append(loss_func(self.model.o3[:, 0], n[0]))
        #print("error", loss_func(self.model.o3[:, 0], n[0]))
        
        # compute the gradient of L wrt non-softmaxed output
        grad_i3 = self.model.o3.copy()
        grad_i3[n, np.arange(len(n))] -= 1

        # compute the gradient of L wrt M3
        grad_M3 = grad_i3 @ self.model.o2.T / Teacher.handful
        # compute the gradient of L wrt b3
        grad_b3 = np.sum(grad_i3, axis=1) / Teacher.handful
        # compute the gradient of L wrt non-activated layer 2 neurons
        grad_i2 = (self.model.M3.T @ grad_i3) * (self.model.o2 != 0)
        
        # compute the gradient of L wrt M2
        grad_M2 = grad_i2 @ self.model.o1.T / Teacher.handful
        # compute the gradient of L wrt b2
        grad_b2 = np.sum(grad_i2, axis=1) / Teacher.handful
        # compute the gradient of L wrt non-activated layer 1 neurons
        grad_i1 = (self.model.M2.T @ grad_i2) * (self.model.o1 != 0)
    
        # compute the gradient of L wrt M1
        grad_M1 = grad_i1 @ picture.T / Teacher.handful
        # compute the gradient of L wrt b1
        grad_b1 = np.sum(grad_i1, axis=1) / Teacher.handful

        # update model with gradient
        self.model.M3 -= Teacher.step * grad_M3
        self.model.b3 -= Teacher.step * grad_b3
        self.model.M2 -= Teacher.step * grad_M2
        self.model.b2 -= Teacher.step * grad_b2
        self.model.M1 -= Teacher.step * grad_M1
        self.model.b1 -= Teacher.step * grad_b1


def loss_func(vec, num):
    return -np.log(vec[num])




dataset = Dataset('.')
dataset.preprocess_images()
model = Network()
teach = Teacher(model, dataset)

for i in range(3):
    teach.train_epoch()

# initialize weights in the neural network using random drawings from a normal distribution with mean 0 and variance 2/784
