import numpy as np, os, sys, pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector



# Input parameters: 
#       LOG_DIR : Full path of log directory. This is where all files will be created.
#       X : Original data
#       feature_vec : Embeddings/ feature vectors learned from model
#       y_class : Target outputs corresponding to input X
#
# Output: 
#       Metadata files will be created in LOG_DIR and a visualization will be created on TensorBoard

def create_visualization(LOG_DIR, X , feature_vec, y_class) :


  # Model checkpoint file. There is no model, but still file will be created.
  path_for_checkpoint = os.path.join(LOG_DIR, "model.ckpt") 

  # Required to load data. TSV format is accepted.
  path_for_metadata =  os.path.join(LOG_DIR,'metadata.tsv') 
  
  # Creates sprites. This displays a the original image during 
  path_for_sprites =  os.path.join(LOG_DIR,'statesimages.png') 

  # Tensor name
  tensor_name = 'color_embeddings'

  
  # Creates a file writer for the log directory
  summary_writer = tf.summary.FileWriter(LOG_DIR)

  # Setup config
  config = projector.ProjectorConfig()
  embedding = config.embeddings.add()

  # Set tensor names and metadata paths
  embedding.tensor_name = tensor_name
  embedding.metadata_path = path_for_metadata 
  
  projector.visualize_embeddings(summary_writer, config)

  # Prepare data in CKPT format
  embedding_var = tf.Variable(feature_vec, name=tensor_name)
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  saver.save(sess, path_for_checkpoint, 1)


  # Prepare metadata in TSV format
  with open(path_for_metadata,'w') as f:
      f.write("Index\tLabel\n")
      for index,label in enumerate(y_class):
          f.write("%d\t%d\n" % (index,label))




  # Run steps :
  # 1. python tensorboard_visualize.py
  # 2. tensorboard --logdir #YOUR_LOG_DIR_PATH# --host=127.0.0.1



'''
# Read input data and feature vectors
f = open(os.getcwd()+"/visualize_data.pkl", 'rb')
X, y = pickle.load(f)
f.close()

# There are 3 classes in my case. Each class has 1450 samples. So, there are 1450*3 = 4350 samples in total.
num_samp = int(X.shape[0]/3)
y_class = [0] * num_samp + [1] * num_samp + [2] * num_samp


LOG_DIR = os.getcwd()+'/color_encoder/'
y = np.copy(X)
y = y.reshape((len(X), 28*28*3))

create_visualization(LOG_DIR, X, y, y_class)
'''
