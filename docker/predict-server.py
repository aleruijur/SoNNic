import sys, logging, os, argparse

import numpy as np
from PIL import Image, ImageGrab
from socketserver import TCPServer, StreamRequestHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and parameters from train module
from train import create_model, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS

# Image preprocessing
def prepare_image(im):
    im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr

# TCP messages handler
class TCPHandler(StreamRequestHandler):
    def handle(self):
        logger.info("Handling a new connection...")
        for line in self.rfile:
            message = str(line.strip(),'utf-8')
            logger.debug(message)

            # Load model when requested by emulator
            if message.startswith("START"):
                weights_file = 'weights/{}.hdf5'.format(args.game)
                logger.info("Loading {}...".format(weights_file))
                model.load_weights(weights_file)

            # Tries to predict image from clipboard, only if emulator is set to work on clipboard mode
            if message.startswith("PREDICTFROMCLIPBOARD"):
                im = ImageGrab.grabclipboard()
                if im != None:
                    prediction = model.predict(prepare_image(im), batch_size=1)[0]
                    self.wfile.write((str(prediction[0]) + "\n").encode('utf-8'))
                else:
                    self.wfile.write("PREDICTIONERROR\n".encode('utf-8'))

            # Tries to predict image from file, usually on TEMP folder
            if message.startswith("PREDICT:"):
                im = Image.open("/server/temp/predict-screenshot.png")
                prediction = model.predict(prepare_image(im), batch_size=1)[0]
                self.wfile.write((str(prediction) + "\n").encode('utf-8'))

if __name__ == "__main__":
    # Parameters grab
    parser = argparse.ArgumentParser(description='Start a prediction server that other apps will call into.')
    parser.add_argument('game', default="demo")
    parser.add_argument('-p', '--port', type=int, help='Port number', default=36296)
    parser.add_argument('-c', '--cpu', action='store_true', help='Force Tensorflow to use the CPU.', default=False)
    args = parser.parse_args()

    # Force to use CPU insted of GPU
    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    #Default model load, no dropout layer
    logger.info("Loading model...")
    model = create_model(keep_prob=1)

    # Starts TCP server on default port 36296
    logger.info("Starting server...")
    server = TCPServer(('0.0.0.0', args.port), TCPHandler)

    print("Listening on Port: {}".format(server.server_address[1]))
    sys.stdout.flush()
    server.serve_forever()
