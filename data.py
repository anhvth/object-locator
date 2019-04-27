import tensorflow as tf
import cv2
import numpy as np

def create_dataset(dataframe, batch_size=2, height=256, width=256):
    needed_columns = ['path', 'locations']
    padded_shapes = ([height, width, 3], [None, 2], [2])
    padding_values = (0., -1., -1)
    output_types = [tf.float32, tf.float32, tf.int32]
    def map_fn(index):
        index = index.numpy()
        path = dataframe.loc[index]['path']
        loc = dataframe.loc[index]['locations']

        img = cv2.imread(path)
        h, w = img.shape[:2]
        hf = h/256
        wf = w/256
        orig_sizes = img.shape[:2]
        orig_sizes = np.array(orig_sizes, dtype=np.int32).reshape([2])
        img = cv2.resize(img, (256, 256))
        img = img / 255
        img = img.astype(np.float32)


        loc = np.array(loc, dtype=np.float32)
        loc[:,0] /= hf
        loc[:,1] /= wf
        #img = np.expand_dims(img, -1)
        return img.astype(np.float32), loc.astype(np.float32), orig_sizes

    for col in needed_columns: assert col in dataframe.keys()

    dataset = tf.data.Dataset.from_tensor_slices(dataframe.index)
    tf_map_fn = lambda path: tf.py_function(map_fn, [path], output_types)
    dataset = dataset.map(tf_map_fn, num_parallel_calls=8)
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=padded_shapes,
                                   padding_values=padding_values)
    return dataset
