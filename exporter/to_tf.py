import argparse
import onnx
from onnx_tf.backend import prepare

def main():
    parser = argparse.ArgumentParser(description='Exports a model to ONNX format.')
    parser.add_argument('model', help='Path to model file')
    args = parser.parse_args()
    
    onnx_model_path = args.model
    print('Importing ONNX model')
    onnx_model = onnx.load(onnx_model_path)
    print('Preparing TF model')
    tf_rep = prepare(onnx_model)
    print(tf_rep)

if __name__ == '__main__':
    main()