import argparse
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img

from train import create_model, INPUT_WIDTH, INPUT_HEIGHT
import matplotlib.pyplot as plt

def drawMap(model_name, img_path):
  weights_file = 'weights/{}.hdf5'.format(model_name)

  model = create_model(keep_prob=1)
  model.load_weights(weights_file)
  successive_outputs = [layer.output for layer in model.layers]
  visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

  img = load_img(img_path, target_size=(INPUT_HEIGHT, INPUT_WIDTH))
  x = img_to_array(img)
  x = x.reshape((1,) + x.shape)
  x /= 255.0
  successive_feature_maps = visualization_model.predict(x)
  layer_names = [layer.name for layer in model.layers]
  for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    print(layer_name, ": ", feature_map.shape)
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

if __name__ == "__main__":
    # Parameters grab
    parser = argparse.ArgumentParser(description='Create a feature map of a model with an image')
    parser.add_argument('model', default="level1")
    parser.add_argument('img_path', default="recordings/record-03f760ee-a79a-42a5-c215-41629ee9f7bf/829.png")
    args = parser.parse_args()

    drawMap(args.model,args.img_path)