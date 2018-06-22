import argparse
import torch, torch.onnx
import constants as const
from model import DeepJ

def main():
    parser = argparse.ArgumentParser(description='Exports a model to ONNX format.')
    parser.add_argument('model', help='Path to model file')
    args = parser.parse_args()

    model = DeepJ()
    # model.load_state_dict(torch.load(args.model))

    evt_input = torch.zeros((1, 1, const.NUM_ACTIONS))
    style_input = torch.zeros((1, const.NUM_STYLES))
    _, states = model.generate(evt_input, style_input, None)
    
    dummy_input = (evt_input, style_input, states)
    torch.onnx.export(model, dummy_input, args.model.replace('.pt', '.onnx'))

if __name__ == '__main__':
    main()