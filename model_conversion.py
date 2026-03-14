# model_conversion.py

import onnx
from onnx_tf.backend import prepare

def convert_model(onnx_model_path):
    model = onnx.load(onnx_model_path)
    tf_rep = prepare(model)
    return tf_rep

# Example usage
# tf_model = convert_model('model.onnx')
