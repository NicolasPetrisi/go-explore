import tensorflow as tf


feature_description = {
    'action': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'reward': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'obs': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)
tf.compat.v1.enable_eager_execution()
raw_dataset = tf.data.TFRecordDataset("000000490704_traj.tfrecords")

parsed_dataset = raw_dataset.map(_parse_function)

i = 0
image_list =[]
for image_features in parsed_dataset:
  image_raw = image_features['obs'].numpy()
  image_list.append(image_raw)
  i += 1
  if i > 1000:
      break
print("length of list is: " + str(len(image_list)))

import cv2
import numpy as np
import glob
import io
import PIL.Image as Image
 
img_array = []


print("before loop")
itr = 0
for byte_img in image_list:
    if len(byte_img) > 0:
        image = Image.open(io.BytesIO(byte_img))
        image.save("images/compressed" +str(itr) + ".png")
        itr += 1

print("between loops")
size = None
for filename in glob.glob('Images/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
print("after loop")
 
out = cv2.VideoWriter('/mnt/c/User/Fredrik/Desctop/output.mp4' ,cv2.VideoWriter_fourcc(*'MP4V'), 15, size)

print("before out loop") 
for i in range(len(img_array)):
    out.write(img_array[i])
print("before realese")
out.release()