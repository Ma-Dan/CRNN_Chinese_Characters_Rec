import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import onnxruntime as rt
import onnx
import onnx.utils as onnxtuils
from onnx.tools import update_model_dims
from onnx import numpy_helper
 
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='images/test.png', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='output/checkpoints/mixed_second_finetune_acc_97P7.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, sess, converter):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    #w_cur = 256.0
    img = cv2.resize(img, (0, 0), fx= w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = np.reshape(img, (1,)+img.shape)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    preds = sess.run([output_name], {input_name: img})[0]
    result = ''
    for i in range(preds.shape[0]):
        idx = np.argmax(preds[i][0])
        if idx != 0:
            result += converter.alphabet[idx-1]

    print('results: {0}'.format(result))

if __name__ == '__main__':
    config, args = parse_arg()

    if False:
        # Change onnx-simplifier processed onnx to dynamic shape
        # Process python -m onnxsim crnn.onnx crnn-sim.onnx --input-shape "1, 1, 32, 168"
        model = onnx.load('crnn-sim.onnx')
        graph = model.graph
        input217 = onnx.helper.make_tensor('217', onnx.TensorProto.INT64, [2], np.array([-1, 512], dtype=int))
        graph.initializer.append(input217)
        input224 = onnx.helper.make_tensor('224', onnx.TensorProto.INT64, [3], np.array([-1, 1, 256], dtype=int))
        graph.initializer.append(input224)
        input370 = onnx.helper.make_tensor('370', onnx.TensorProto.INT64, [2], np.array([-1, 512], dtype=int))
        graph.initializer.append(input370)
        input377 = onnx.helper.make_tensor('377', onnx.TensorProto.INT64, [3], np.array([-1, 1, 6736], dtype=int))
        graph.initializer.append(input377)
        onnx.save(model, 'crnn-dynamic.onnx')

    sess = rt.InferenceSession("crnn-dynamic.onnx")

    started = time.time()

    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    recognition(config, img, sess, converter)

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))

