import argparse
import torch, torch.onnx
import constants as const
from model import DeepJ

def main():
    parser = argparse.ArgumentParser(description='Exports a model to ONNX format.')
    parser.add_argument('model', help='Path to model file')
    args = parser.parse_args()

    model = DeepJ()
    model.load_state_dict(torch.load(args.model))

    dummy_input = (
        # Event
        torch.zeros((1, 1, const.NUM_ACTIONS)),
        # Style
        torch.zeros((1, const.NUM_STYLES)),
        # Memory states
        (torch.zeros((2, 1, model.num_units)),) + 
        tuple(
            (torch.tensor(0), tuple(torch.zeros((1, 1, model.num_units)) for _ in range(model.rnns[l + 1].dilation)))
            for l in range(model.num_layers - 1)
        )
    )
    torch.onnx.export(model, dummy_input, args.model.replace('.pt', '.onnx'))

if __name__ == '__main__':
    main()