import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img

from train import create_model, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS
import matplotlib.pyplot as plt

img_path=r"recordings/record-3b6f209d-92ab-46d8-c6b5-9fa77fda770a/454.png"
#"recordings/record-3b6f209d-92ab-46d8-c6b5-9fa77fda770a/183.png 454"
#"recordings\record-2d61d69a-66c2-4d2b-cddc-ea39c0c1ac97\396.png"
weights_file = 'weights/{}.hdf5'.format("lowConv")

model = create_model(keep_prob=1)
#model.summary()
model.load_weights(weights_file)
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

img = load_img(img_path, target_size=(INPUT_HEIGHT, INPUT_WIDTH))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
x /= 255.0
successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers]
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  print(feature_map.shape)
  if len(feature_map.shape) == 4:
    n_features = feature_map.shape[-1]
    size_x = feature_map.shape[1]
    size_y = feature_map.shape[2]

    display_grid = np.zeros((size_x, size_y * n_features))
    
    for i in range(n_features):
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size_y : (i + 1) * size_y] = x
    
    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' )
plt.show()