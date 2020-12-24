from model import Generator
import tensorflow as tf
from skimage import io
multiplier = 1
noisesize = 67
generator = Generator(64*multiplier, 8, 3)
generator.load_weights("brickwall")
zl = tf.random.uniform([1, noisesize, noisesize, generator.dl])
zg = tf.random.uniform([1, 1, 1, generator.dg])
zgs = tf.concat([tf.concat([zg for i in range(noisesize)],axis=1) for j in range(noisesize)],axis=2)
train_batch = tf.concat([zl, zgs], axis=3)
zg = tf.reshape(zg, [1, generator.dg])
outimg = tf.reshape(generator(train_batch, zg, training=False), [noisesize*32-124,noisesize*32-124,3])
io.imsave("generated.jpg", outimg)
