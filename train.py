import tensorflow as tf
import losses
import data
from models import unet_model
import pandas
import logging
import os
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', choices=['train', 'debug'], 
		help='if mode is debug then init small net with 1 batch example to test')
parser.add_argument('--height', default=256, type=int, help='image height')
parser.add_argument('--width', default=256, type=int, help='image width')
parser.add_argument('--lr', default=4e-4, help='learning rate')
parser.add_argument('--p', default=-3, help='UNKNOWN ')
parser.add_argument('--max_epochs', default=int(10e5), help='num of max epoch ')
parser.add_argument('--output_dir', default='training_checkpoints', help='directory to save model weights')
parser.add_argument('--nClass', default=1, help='num of output channels')
parser.add_argument('--batch_size', default=2, help='batch size')
parser.add_argument('--max_to_keep', default=2, help='batch size')



args = parser.parse_args()

logger = logging.getLogger()
print('version: {}'.format(tf.__version__))

if __name__ == "__main__":
    output_dir = os.path.join(args.output_dir, 'samples_images')
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = unet_model.UNet(args.nClass, args.height, args.width)
    loss_loc = losses.WeightedHausdorffDistance(
        args.height, args.width, p=args.p, return_2_terms=True)
    # dataset
    df = pandas.read_json('./datasets/train/train.json')
    # df = df[:args.batch_size]
    dataset = data.create_dataset(df, args.batch_size, args.height, args.width)
    dataset = dataset.prefetch(100)
    iter = dataset.repeat(args.max_epochs).make_one_shot_iterator()

    imgs, locs, orig_sizes = iter.get_next()
    # train ops
    prob_map = model(imgs)
    optimizer = tf.train.AdamOptimizer(args.lr, use_locking=True)

    term1, term2 = loss_loc(prob_map, locs, orig_sizes)
    loss = term1 + term2
    global_step = tf.train.get_global_step()

    train_step = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=args.max_to_keep)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    steps_per_epoch = len(df)// args.batch_size

    for epoch in range(args.max_epochs):
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


