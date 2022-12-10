import glob
import numpy as np
import os.path
import tensorflow as tf


def embedding_resnet(X_PATH,Y_PATH):
  #====define model
  X_input=tf.keras.Input(shape=(64,64,3))
  output=tf.keras.applications.ResNet50V2(weights="imagenet",pooling=None,include_top=False)(X_input)
  model=tf.keras.Model(inputs=X_input,outputs=output)
  model.summary()

  #====load dataset
  X_TEST=np.load(X_PATH)
  Y_TEST=np.load(Y_PATH)

  X_TEST= tf.cast(X_TEST, tf.float32) 
  Y_TEST= tf.cast(Y_TEST, tf.float32) 


  embedding_resnet = model.predict(X_TEST)
  print(embedding_resnet.shape)
  embedding_resnet_r=tf.reshape(embedding_resnet,shape=(embedding_resnet.shape[0],-1))
  Y_TEST = tf.expand_dims(Y_TEST, 1)
  embedding_final=tf.concat([Y_TEST, embedding_resnet_r], axis=1)
  index=tf.squeeze(embedding_final[:,0:1])
  index= tf.cast(index, tf.int32) 
  num_class=len(np.unique(index))
  class_mean=tf.math.unsorted_segment_mean(embedding_final, index,num_class)
  print(class_mean.shape)
  return class_mean


#=====cat tune and the TL dataset embedding
if __name__ == "__main__":
  X_PATH="/content/drive/My Drive/cs330_dataset/x1_64.npy"
  Y_PATH="/content/drive/My Drive/cs330_dataset/y1_64.npy"
  x_tune=np.load("/content/drive/My Drive/cs330_dataset/x_tune_64_all_embedding_class_mean.npy")
  class_mean_x=embedding_resnet(X_PATH,Y_PATH)
  out=tf.concat([class_mean_x,x_tune],axis=0)
  print(out.shape)
  np.save("/content/drive/My Drive/cs330_dataset/x1_64_all_embedding_class_mean.npy",out)