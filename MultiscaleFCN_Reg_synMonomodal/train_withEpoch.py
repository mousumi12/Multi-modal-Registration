import tensorflow as tf
from models import DIRNet
from config import get_config
from data import MNISTDataHandler
from ops import mkdir
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3,2" #model will be trained on GPU 0


best_validation_loss = 9999

def main():
  
  global best_validation_loss

  sess = tf.Session()
  config = get_config(is_train=True)
  mkdir(config.tmp_dir)
  mkdir(config.ckpt_dir1)
  mkdir(config.ckpt_dir2)

  reg = DIRNet(sess, config, "DIRNet", is_train=True)
  dh = MNISTDataHandler(is_train=True)

  config.train_iter_num = 30530//config.batch_size  #len(os.listdir(train_x_dir)) // config.batch_size
  config.val_iter_num = 7634//config.batch_size    # total no of data 38164

  for epoch in range(config.epoch_num):
    train_L = []
    for i in range(config.train_iter_num):   #(config.iteration):
      batch_x_train, batch_y_train, name1, batch_x_val, batch_y_val, name2 = dh.sample_pair(config.batch_size)
      #print(np.shape(batch_x))
      #print(np.shape(batch_y))

      loss = reg.fit(batch_x_train, batch_y_train)
      train_L.append(loss)
      print("iter {:>6d} : {}".format(i+1, loss))
    print('[TRAIN] epoch={:>3d}, loss={:.4f}.'.format(epoch + 1, sum(train_L)/len(train_L))) 

    valid_L = []
    for j in range(config.val_iter_num):
      batch_x_train, batch_y_train, name1, batch_x_val, batch_y_val, name2 = dh.sample_pair(config.batch_size)
      loss_valid = reg.deploy(config.tmp_dir, batch_x_val, batch_y_val, name2)
      valid_L.append(loss_valid)
      print("iter {:>6d} : {}".format(j+1, loss_valid))
    print('[VALID] epoch={:>3d}, loss={:.4f}.'.format(epoch + 1, sum(valid_L) / len(valid_L)))

    if (epoch + 1) % config.save_interval == 0:   #(i+1) % 1000 == 0:
        batch_x_train, batch_y_train, name1, batch_x_val, batch_y_val, name2 = dh.sample_pair(config.batch_size)
        loss_valid = reg.deploy(config.tmp_dir, batch_x_val, batch_y_val, name2)
        #print("iter {:>6d} : {}".format(i+1, loss_valid))
        #if loss_valid < best_validation_loss:
        #best_validation_loss = loss_valid
        reg.save(config.ckpt_dir1, config.ckpt_dir2)
        #else:
        #  print("Current validation loss {} did not improved from:{}".format(loss_valid, best_validation_loss))
      

if __name__ == "__main__":
  main()
