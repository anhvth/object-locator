import tensorflow as tf
import losses
import data
from models import unet_model
import pandas
import logging
import os
import cv2

logger = logging.getLogger()
#logger.setLevel(level=logging.DEBUG)
print('version: {}'.format(tf.__version__))

if __name__ == "__main__":
    nClass = 1
    height = 256
    width = 256
    batch_size = 2
    p = -3
    lr = 4e-4
    max_epochs = int(10e5)
    output_dir = 'samples_images/'
    checkpoint_dir = 'checkpoints/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = unet_model.UNet(nClass, height, width)
    loss_loc = losses.WeightedHausdorffDistance(
        height, width, p=p, return_2_terms=True)
    # dataset
    df = pandas.read_json('./datasets/train/train.json')
    # df = df[:batch_size]
    dataset = data.create_dataset(df, batch_size, height, width)
    dataset = dataset.prefetch(100)
    iter = dataset.repeat(max_epochs).make_one_shot_iterator()

    imgs, locs, orig_sizes = iter.get_next()
    # train ops
    prob_map = model(imgs)
    optimizer = tf.train.AdamOptimizer(lr, use_locking=True)

    term1, term2 = loss_loc(prob_map, locs, orig_sizes)
    loss = term1 + term2
    global_step = tf.train.get_global_step()

    train_step = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=2)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    steps_per_epoch = len(df)// batch_size

    for epoch in range(max_epochs):
        for step in range(steps_per_epoch):
            _imgs, _prob_map, _t1, _t2, _= sess.run([imgs, prob_map, term1, term2, train_step])
            print('{}:{}-Term1: {:0.5f}, Term2:{:0.5f}'.format(epoch, step, _t1, _t2))
            g_step = epoch*(steps_per_epoch)+step
            if g_step % 200 == 0:
                _prob_map_0 = (_prob_map[0]*255).astype('uint8')
                _img_0 = _imgs[0]
                _img_0 = _img_0*255
                _img_0 = _img_0.astype('uint8')
                _img_0[...,0] = _prob_map_0
                name = '{}-{}.png'.format(epoch, step)
                cv2.imwrite(os.path.join(output_dir, name), _img_0)

        saver.save(sess, checkpoint_dir+'unet', global_step=epoch)


