import torch
from network import MyNet
from metric import valid
import argparse
from dataloader import load_data
import os


def str_to_list(s):
    return [int(x) for x in s.split(',')]

Dataname = 'UCIDigit'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument("--feature_dim", default=1024)
parser.add_argument('--h_dims', default=[1024,1024,1024], type=str_to_list)
parser.add_argument('--fusion_indices', default=[0,1,2,3], type=str_to_list)
parser.add_argument('--steps', default=3, type=int)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--temperature', default=0.1, type=float)
parser.add_argument('--model_path', default=None)

args = parser.parse_args()

if torch.cuda.is_available():
    target_device = 1
    torch.cuda.set_device(target_device)
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(target_device)}")
else:
    print("GPU not found!")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*60)
print(args)
print("="*60)

dataset, dims, view, data_size, class_num = load_data(args.dataset)

def run_clustering_test():
    model_path = args.model_path
    if model_path is None:
        model_path = os.path.join("models", f"{args.dataset}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model parameters not found: {model_path}")

    model = MyNet(dims, args.feature_dim, view, args.h_dims, args.fusion_indices, args.steps, args.alpha, args.temperature, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Model parameters loaded from: {model_path}")
    acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num)

    print(f"\n====== Clustering Test Results ======")
    print(f"ACC: {acc:.4f} | NMI: {nmi:.4f} | PUR: {pur:.4f} | ARI: {ari:.4f}")

if __name__ == '__main__':
    run_clustering_test()
