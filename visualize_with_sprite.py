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
  embedding.sprite.image_path = path_for_sprites
  
  # Sprite image size
  embedding.sprite.single_image_dim.extend([28,28,3])
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



  # Prepare image sprite in png format
  to_visualise = X
  
  # Reshapes images shape to (batch,28,28,3) for color images and (batch, 28, 28) to grayscale images.
  to_visualise = np.reshape(to_visualise,(-1,28,28,3))

  # Invert image
  to_visualise = 1-to_visualise

  to_visualise = np.array(to_visualise)
  img_h = to_visualise.shape[1]
  img_w = to_visualise.shape[2]
  n_plots = int(np.ceil(np.sqrt(to_visualise.shape[0])))

  # Create sprite template
  sprite_image = np.ones((img_h * n_plots ,img_w * n_plots, 3 ))
  
  # Fill the sprite templates with the input images
  for i in range(n_plots):
      for j in range(n_plots):
          this_filter = i * n_plots + j
          if this_filter < to_visualise.shape[0]:
              this_img = to_visualise[this_filter]
              sprite_image[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w,:] = this_img
              
  # Save the sprite image
  plt.imsave(path_for_sprites,sprite_image)
  plt.imshow(sprite_image)

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
# sys.exit(1)
create_visualization(LOG_DIR, X, y, y_class)
'''
