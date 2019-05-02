from tqdm import tqdm
import tensorflow as tf
import losses
import data
from models import unet_model, discriminator
import pandas
import logging
import os
import cv2
import argparse
import summary

print('version: {}'.format(tf.__version__))

os.environ['KMP_DUPLICATE_LIB_OK']='True' # remove bug in macos https://github.com/dmlc/xgboost/issues/1715

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', choices=['train', 'debug'], 
		help='if mode is debug then init small net with 1 batch example to test')
parser.add_argument('--height', default=256, type=int, help='image height')
parser.add_argument('--width', default=256, type=int, help='image width')
parser.add_argument('--glr', default=4e-4, help='learning rate')
parser.add_argument('--dlr', default=4e-4, help='learning rate')
parser.add_argument('--p', default=-3, help='UNKNOWN ')
parser.add_argument('--max_epochs', default=int(10e5), help='num of max epoch ')
parser.add_argument('--output_dir', default='training_checkpoints', help='directory to save model weights')
parser.add_argument('--nClass', default=1, help='num of output channels')
parser.add_argument('--batch_size', default=1, help='batch size')
parser.add_argument('--max_to_keep', default=2, help='max keeping checkpoint')
parser.add_argument('--ngf', type=int, default=64, help='init num of filter for unet')
parser.add_argument('--summary_freq', type=int, default=20, help='scalars summary frequence')
parser.add_argument('--display_freq', type=int, default=100, help='image summary frequence')
parser.add_argument('--ndf', type=int, default=64, help='init num of filter for discriminator')



args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

if __name__ == "__main__":
    if args.mode == 'debug':
        args.ngf = 8
        args.ndf = 8
        # df = df[:args.batch_size]
        args.display_freq = 5
        args.summary_freq = 2

    print('---- hyper parameters>>>>>>>')
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))

    print('<<<<<< hyper parameters')

    output_dir = os.path.join(args.output_dir, 'samples_images')
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    summary_dir = os.path.join(args.output_dir, 'summary')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    



    loss_loc = losses.WeightedHausdorffDistance(
        args.height, args.width, p=args.p, return_2_terms=True)
   # create dataset
    df = pandas.read_json('./datasets/train/train.json')

    dataset = data.create_dataset(df, args.batch_size, args.height, args.width)
    dataset = dataset.prefetch(100)
    iter = dataset.repeat(args.max_epochs).make_one_shot_iterator()

    inputs, locs, tar_prob_maps = iter.get_next()

    # create model
    discrim_model = discriminator.Discriminator(args.ndf)

    model = unet_model.UNet(args.nClass, args.height, args.width, ngf=args.ngf)
    with tf.variable_scope('generator'):
        prob_map = model(inputs)
        prob_map.set_shape([None, args.height, args.width])
    # create train ops
    # create adversarial loss

    pred_prob_maps = tf.expand_dims(prob_map, axis=-1)
    with tf.variable_scope('discriminator'):
        pred_real = discrim_model(inputs, tar_prob_maps)
        pred_fake = discrim_model(inputs, pred_prob_maps)


    gen_loss_gan = -tf.log(pred_fake)

    dis_loss_real = -tf.log(pred_real)
    dis_loss_fake = -tf.log(0.9-pred_fake)
    dis_loss = dis_loss_real + dis_loss_fake

    g_opt = tf.train.AdamOptimizer(args.glr, use_locking=True)
    d_opt = tf.train.AdamOptimizer(args.dlr)
    term_1, term_2, term_3 = loss_loc(prob_map, locs)

    tf.summary.scalar('Term_1', term_1)
    tf.summary.scalar('Term_2', term_2)
    tf.summary.scalar('Term_3', term_3)
    tf.summary.image('Gen_loss_gan', gen_loss_gan)
    tf.summary.image('Dis_loss_gan', dis_loss)

    g_loss = term_1 + term_2 + term_3 + gen_loss_gan
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')] 
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')] 
    print('Num of G-var: {}'.format(len(g_vars)))
    print('Num of D-var: {}'.format(len(d_vars)))

    g_grads_vars = g_opt.compute_gradients(g_loss, var_list=g_vars)
    g_mean_grad = summary.get_mean_grad([gv[0] for gv in g_grads_vars])


    d_grads_vars = g_opt.compute_gradients(dis_loss, var_list=d_vars)
    d_mean_grad = summary.get_mean_grad([gv[0] for gv in d_grads_vars])

    tf.summary.scalar('G-mean_grad', g_mean_grad)
    tf.summary.scalar('D-mean_grad', d_mean_grad)

    merge_op = tf.summary.merge_all()

    summary_image = [inputs[...,0 ], inputs[...,1], inputs[..., 2]*.3+prob_map*.7]

    summary_image = tf.stack(summary_image, axis=-1)
    summary_image_op = tf.summary.image('train-predict', summary_image)

    g_step = g_opt.apply_gradients(g_grads_vars, global_step=global_step)
    d_step = g_opt.apply_gradients(d_grads_vars)

    saver = tf.train.Saver(max_to_keep=args.max_to_keep)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    steps_per_epoch = len(df)// args.batch_size

    train_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # training loop
    for epoch in range(args.max_epochs):
        pbar = tqdm(range(steps_per_epoch))
        for step in pbar: 
            gstep = sess.run(global_step)
            train_dict = { 
                            'term_1':term_1, 
                            'term_2':term_2,
                            'term_3':term_3,
                            'dis_loss':dis_loss,
                            'gen_loss':gen_loss_gan,
                            'g_step':g_step,
                            'd_step':d_step,
                        }
            if gstep % args.summary_freq == 0:
                train_dict['merge_op'] = merge_op

            if gstep % args.display_freq == 0:
                train_dict['summary_image_op'] = summary_image_op
            run_results = sess.run(train_dict)
            description = 'Epoch {}:{}-Term1: {:0.3f}, Term2:{:0.3f}, \
                    Term2:{:0.3f}, Gen:{:0.3f}, Dis:{:0.3f}'.format(epoch, 
                                                                    step,
                                                                    run_results['term_1'],
                                                                    run_results['term_2'],
                                                                    run_results['term_3'],
                                                                    run_results['gen_loss'].mean(),
                                                                    run_results['dis_loss'].mean(),
                                                                    )
            pbar.set_description(description)
            if gstep % args.summary_freq == 0:
                train_writer.add_summary(run_results['merge_op'], gstep)

            if gstep % args.display_freq == 0:
                train_writer.add_summary(run_results['summary_image_op'], gstep)

        saver.save(sess, checkpoint_dir+'unet', global_step=epoch)


