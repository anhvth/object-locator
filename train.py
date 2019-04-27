from tqdm import tqdm
import tensorflow as tf
import losses
import data
from models import unet_model
import pandas
import logging
import os
import cv2
import argparse
import summary


os.environ['KMP_DUPLICATE_LIB_OK']='True' # remove bug in macos https://github.com/dmlc/xgboost/issues/1715

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
parser.add_argument('--max_to_keep', default=2, help='max keeping checkpoint')
parser.add_argument('--ngf', type=int, default=64, help='init num of filter for unet')
parser.add_argument('--summary_freq', type=int, default=20, help='scalars summary frequence')
parser.add_argument('--display_freq', type=int, default=100, help='image summary frequence')



args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)
print('version: {}'.format(tf.__version__))

if __name__ == "__main__":
        
    if args.mode == 'debug':
        args.ngf = 8
        # df = df[:args.batch_size]
        args.display_freq = 5
        args.summary_freq = 2

    output_dir = os.path.join(args.output_dir, 'samples_images')
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    summary_dir = os.path.join(args.output_dir, 'summary')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    model = unet_model.UNet(args.nClass, args.height, args.width, ngf=args.ngf)
    loss_loc = losses.WeightedHausdorffDistance(
        args.height, args.width, p=args.p, return_2_terms=True)
    # dataset
    df = pandas.read_json('./datasets/train/train.json')
    
    print('---- hyper parameters>>>>>>>')

    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))

    print('<<<<<< hyper parameters')

    dataset = data.create_dataset(df, args.batch_size, args.height, args.width)
    dataset = dataset.prefetch(100)
    iter = dataset.repeat(args.max_epochs).make_one_shot_iterator()

    imgs, locs, orig_sizes = iter.get_next()
    # train ops
    with tf.variable_scope('unet'):
        prob_map = model(imgs)

    prob_map.set_shape([None, args.height, args.width])

    optimizer = tf.train.AdamOptimizer(args.lr, use_locking=True)

    term1, term2 = loss_loc(prob_map, locs, orig_sizes)
    
    tf.summary.scalar('Term_1', term1)
    tf.summary.scalar('Term_2', term2)

    loss = term1 + term2
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    train_vars = [v for v in tf.trainable_variables() if v.name.startswith('unet')] 
    print('Num of train var: {}'.format(len(train_vars)))
    grads_vars = optimizer.compute_gradients(loss, var_list=train_vars)
    mean_grad = summary.get_mean_grad([gv[0] for gv in grads_vars])
    tf.summary.scalar('mean_grad', mean_grad)
    merge_op = tf.summary.merge_all()


    # summary output image
    # import ipdb; ipdb.set_trace()
    summary_image = [imgs[...,0 ], imgs[...,1], imgs[..., 2]*.3+prob_map*.7]

    summary_image = tf.stack(summary_image, axis=-1)
    summary_image_op = tf.summary.image('train-predict', summary_image)

    train_step = optimizer.apply_gradients(grads_vars, global_step=global_step)


    saver = tf.train.Saver(max_to_keep=args.max_to_keep)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    steps_per_epoch = len(df)// args.batch_size

    train_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # import ipdb; ipdb.set_trace()
    # training loop
    for epoch in range(args.max_epochs):
        pbar = tqdm(range(steps_per_epoch))
        for step in pbar: 
            gstep = sess.run(global_step)
            train_dict = { 
                            'term1':term1, 
                            'term2':term2, 
                            'train_step':train_step,
                        }
            if gstep % args.summary_freq == 0:
                train_dict['merge_op'] = merge_op

            if gstep % args.display_freq == 0:
                train_dict['summary_image_op'] = summary_image_op
            run_results = sess.run(train_dict)
            description = 'Epoch {}:{}-Term1: {:0.5f}, Term2:{:0.5f}'.format(epoch, 
                                                                    step,
                                                                    run_results['term1'],
                                                                    run_results['term2'])
            pbar.set_description(description)
            if gstep % args.summary_freq == 0:
                train_writer.add_summary(run_results['merge_op'], gstep)

            if gstep % args.display_freq == 0:
                train_writer.add_summary(run_results['summary_image_op'], gstep)

        saver.save(sess, checkpoint_dir+'unet', global_step=epoch)


