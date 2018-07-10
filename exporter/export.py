"""
Exports to ONNX model format
"""
import argparse
import torch#, torch.onnx
from webdnn.frontend.pytorch import PyTorchConverter
import constants as const
from model import DeepJ

def main():
    parser = argparse.ArgumentParser(description='Exports a model to ONNX format.')
    parser.add_argument('model', help='Path to model file')
    args = parser.parse_args()

    onnx_model_path = args.model.replace('.pt', '.onnx')

    print('Loading Pytorch model')
    model = DeepJ()
    model.load_state_dict(torch.load(args.model))

    evt_input = torch.zeros((1, 1, const.NUM_ACTIONS))
    style_input = torch.zeros((1, const.NUM_STYLES))
    _, states = model.generate(evt_input, style_input, None)
    
    dummy_input = (evt_input, style_input, states)

    print('Generating graph')
    graph = PyTorchConverter().convert(model, dummy_input)
    # print(graph)
    # torch.onnx.export(model, dummy_input, onnx_model_path)
    # print('Exported to ONNX format')

if __name__ == '__main__':
    main()