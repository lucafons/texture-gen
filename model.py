from tensorflow.keras.layers import Dense, Layer
import tensorflow as tf
import numpy as np
import os
import math as m
from skimage import io
from random import randint
os.environ['DGLBACKEND'] = "tensorflow"

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (5,5), strides=(2,2), padding="same",activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.conv2 = tf.keras.layers.Conv2D(64, (5,5), strides=(2,2), padding="same",activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.conv3 = tf.keras.layers.Conv2D(128, (5,5), strides=(2,2), padding="same",activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.conv4 = tf.keras.layers.Conv2D(256, (5,5), strides=(2,2), padding="same",activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.conv5 = tf.keras.layers.Conv2D(1, (5,5), strides=(2,2), padding="same")
        self.leakyrelu = tf.keras.layers.LeakyReLU()
    def call(self, image):
        normalization = tf.keras.layers.BatchNormalization()
        dropoutn = 0.3
        layer1 = self.conv1(image)
        layer1 = tf.keras.layers.Dropout(dropoutn)(layer1)

        layer2 = self.conv2(layer1)
        layer2 = tf.keras.layers.Dropout(dropoutn)(layer2)

        layer3 = self.conv3(layer2)
        layer3 = tf.keras.layers.Dropout(dropoutn)(layer3)

        layer4 = self.conv4(layer3)
        layer4 = tf.keras.layers.Dropout(dropoutn)(layer4)

        layer5 = self.conv5(layer4)
        return layer5
    def loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
        #return -tf.reduce_sum(tf.math.log(real_output)+tf.math.log(1-fake_output))
class Generator(tf.keras.Model):
    def __init__(self, dl, dg, dp):
        super(Generator, self).__init__()
        self.dl = dl
        self.dg = dg
        self.dp = dp
        self.periodmlp = PeriodMLP(dg, dp)
        self.batch_size = 16
        #self.conv1 = tf.keras.layers.Conv2D(256, 5, 1, padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.2), use_bias=False)
        #self.conv2 = tf.keras.layers.Conv2D(128, 5, 1, padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.2), use_bias=False)
        #self.conv3 = tf.keras.layers.Conv2D(64, 5, 1, padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.2), use_bias=False)
        #self.conv4 = tf.keras.layers.Conv2D(32, 5, 1, padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.2), use_bias=False)
        #self.conv5 = tf.keras.layers.Conv2D(3, 5, 1, padding="valid", activation="tanh", use_bias=False)
        self.conv1 =  tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.conv2 =  tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.conv3 =  tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.conv4 =  tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.conv5 =  tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.normalization1 = tf.keras.layers.BatchNormalization()
        self.normalization2= tf.keras.layers.BatchNormalization()
        self.normalization3 = tf.keras.layers.BatchNormalization()
        self.normalization4 = tf.keras.layers.BatchNormalization()

    def call(self, image, zg):
        L = image.shape[1]
        M = image.shape[2]
        dropoutn = 0.0
        upsample = tf.keras.layers.UpSampling2D()
        zmatr = tf.transpose(self.periodmlp(zg),perm=[0,2,1])
        ptensor = np.zeros([L,M,self.dp])
        phii = tf.random.uniform([self.dp,image.shape[0]], maxval=2*m.pi)
        gammam = np.zeros((L,M))
        for i in range(L):
            gammam[i] = i+1
        mum = np.transpose(gammam)
        arrs = []
        for i in range(self.dp):
            a,b = zmatr[:,i][:,0],zmatr[:,i][:,1]
            arrs.append(a[:,None,None]*gammam+b[:,None,None]*mum+phii[i][:,None,None])
        arrs = [tf.reshape(arr, [image.shape[0],L,M,1]) for arr in arrs]
        ptensor = tf.concat(arrs,axis=3)
        ptensor = tf.sin(ptensor)
        #image = tf.concat([image,ptensor], axis=3)
        layer1 = upsample(image)
        layer1 = self.conv1(layer1)
        layer1 = tf.keras.layers.Dropout(dropoutn)(layer1)
        layer1 = self.normalization1(layer1)

        layer2 = upsample(layer1)
        layer2 = self.conv2(layer2)
        layer2 = tf.keras.layers.Dropout(dropoutn)(layer2)
        layer2 = self.normalization2(layer2)

        layer3 = upsample(layer2)
        layer3 = self.conv3(layer3)
        layer3 = tf.keras.layers.Dropout(dropoutn)(layer3)
        layer3 = self.normalization3(layer3)

        layer4 = upsample(layer3)
        layer4 = self.conv4(layer4)
        layer4 = tf.keras.layers.Dropout(dropoutn)(layer4)
        layer4 = self.normalization4(layer4)

        layer5 = upsample(layer4)
        layer5 = self.conv5(layer5)

        return layer5
    def loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)
        #return tf.reduce_sum(tf.math.log(1-fake_output))
class PeriodMLP(Layer):

    def __init__(self, dg, dp):

        super(PeriodMLP, self).__init__()
        self.dg = dg
        self.dp = dp
        self.dh = 60
        self.W = tf.Variable(tf.random.normal([self.dh,dg], stddev=.1))
        self.W1 = tf.Variable(tf.random.normal([dp,self.dh], stddev=.1))
        self.W2 = tf.Variable(tf.random.normal([dp,self.dh], stddev=.1))
        self.b = tf.Variable(tf.random.normal([self.dh,1], stddev=.1))
        self.b1 = tf.Variable(tf.random.normal([dp,1], stddev=.1))
        self.b2 = tf.Variable(tf.random.normal([dp,1], stddev=.1))

    def call(self, zg):
        zg = tf.reshape(zg, [zg.shape[0],zg.shape[1],1])
        row1 = tf.nn.relu(tf.matmul(self.W, zg)+self.b)
        row2 = tf.nn.relu(tf.matmul(self.W, zg)+self.b)
        row1 = tf.transpose(tf.matmul(self.W1, row1)+self.b1,perm=[0,2,1])
        row2 = tf.transpose(tf.matmul(self.W2, row2)+self.b2,perm=[0,2,1])
        return tf.concat([row1,row2],axis=1)
def gen_image(generator, x, y):
    L, M = int(x/32), int(y/32)
    zl = tf.random.uniform([1, L, M, 3])
    zg = tf.random.uniform([1, 1, 1, generator.dg])
    zgs = tf.concat([tf.concat([zg for i in range(L)],axis=1) for j in range(M)],axis=2)
    train_batch = tf.concat([zl, zgs], axis=3)
    zg = tf.reshape(zg, [1, generator.dg])
    outimg = tf.reshape(generator(train_batch, zg, training=False), [x,y,3])
    return outimg
#@tf.function
def train_step(images, generator, discriminator):
    L = int(images.shape[1]/32)
    M = int(images.shape[2]/32)
    zl = tf.random.uniform([images.shape[0], L, M, 3])
    zg = tf.random.uniform([images.shape[0], 1, 1, generator.dg])
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    zgs = tf.concat([tf.concat([zg for i in range(L)],axis=1) for j in range(M)],axis=2)
    train_batch = tf.concat([zl, zgs], axis=3)
    zg = tf.reshape(zg, [images.shape[0], generator.dg])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(train_batch, zg, training=True)

        real = discriminator(images)
        fake = discriminator(generated)
        real = tf.reshape(real, [images.shape[0], -1])
        fake = tf.reshape(fake, [images.shape[0], -1])
        genloss = generator.loss(fake)
        discloss = discriminator.loss(real, fake)
    gradients_of_generator = gen_tape.gradient(genloss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(discloss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print("Discriminator loss", discloss)
    print("Generator loss", genloss)
def train(generator, discriminator, image):
    epochs = 100
    numtrain = 4000
    images = []
    X = image.shape[0]
    Y = image.shape[1]
    patchsize = 192
    for k in range(numtrain):
        xstart = randint(0, X-patchsize-1)
        ystart = randint(0, Y-patchsize-1)
        images.append(image[xstart:xstart+patchsize,ystart:ystart+patchsize])
    images = tf.concat([tf.reshape(im, [1, patchsize, patchsize, 3]) for im in images], axis=0)
    images = tf.cast(images, tf.float32)/255
    for i in range(epochs):
        for j in range(int(numtrain/16)):
            print(i, j)
            train_step(images[j*16:(j+1)*16], generator, discriminator)
        zl = tf.random.uniform([1, 8, 8, 3])
        zg = tf.random.uniform([1, 1, 1, generator.dg])
        zgs = tf.concat([tf.concat([zg for i in range(8)],axis=1) for j in range(8)],axis=2)
        train_batch = tf.concat([zl, zgs], axis=3)
        zg = tf.reshape(zg, [1, generator.dg])
        outimg = tf.reshape(generator(train_batch, zg, training=False), [256,256,3])
        #outimg = (outimg-np.min(outimg))/(np.max(outimg)-np.min(outimg))
        #outimg *= 255
        #outimg = tf.cast(outimg, tf.uint8)
        #outimg = (outimg+1)*128-1
        #outimg = tf.cast(outimg, tf.uint8)
        io.imsave(str(i)+".jpg", outimg)
    io.imsave("final.jpg", gen_image(generator(1024,1024)))
def main():
    model = Generator(128, 128, 3)
    discriminator = Discriminator()
    image = io.imread("stonebricks.jpg")
    train(model, discriminator, image)
if __name__ == '__main__':
    main()
