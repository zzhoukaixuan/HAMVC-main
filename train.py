import torch
from network import MyNet
from metric import valid
import argparse
from loss import Loss
from dataloader import load_data
import os
import time


def str_to_list(s):
    return [int(x) for x in s.split(',')]

Dataname = 'UCIDigit'  # 'ALOI100'  'Animal' 'handwritten' 'UCIDigit' 'Scene15' 'NoisyMNIST'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--feature_dim", default=1024)
parser.add_argument('--h_dims', default=[1024,1024,1024], type=str_to_list)
parser.add_argument('--fusion_indices', default=[0,1,2,3], type=str_to_list)
parser.add_argument('--steps', default=3, type=int)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--temperature', default=0.1, type=float)
parser.add_argument('--lambda1', default=0.1, type=float)
parser.add_argument('--lambda2', default=0.1, type=float)

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

dataset_configs = {
    "NoisyMNIST": (10, 1),
    "UCIDigit": (500, 0),
    "handwritten": (450, 0),
    "ALOI100": (50, 3),
    "Animal": (60, 3),
    "Cifar10":(10, 1),
    "Scene15":(400, 5),
    "MNIST":(2, 0),
}

if args.dataset in dataset_configs:
    args.mse_epochs, seed = dataset_configs[args.dataset]
else:
    seed = 5

print("="*60)
print(args)
print("="*60)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

if not os.path.exists('./models'):
    os.makedirs('./models')

def train_one_epoch(model, optimizer, criterion, epoch, lambda_uniformity=1.0, lambda_alignment=1.0):
    model.train()
    tot_loss = 0.
    tot_recon = 0.
    tot_inter = 0.
    view_pairs = view * (view - 1) // 2
    mse = torch.nn.MSELoss()

    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()       
        xrs, zs, _ = model(xs)
        recon_list ,inter_list= [], []   
        for v in range(view):
            recon_list.append(mse(xs[v], xrs[v]))
            for w in range(v+1, view):
                inter_list.append(criterion.positive_contrastive_loss(
                                    zs[v], zs[w], 
                                    lambda_uniformity=lambda_uniformity, 
                                    lambda_alignment=lambda_alignment))
                
        recon = sum(recon_list) / view
        inter = sum(inter_list) / view_pairs
        loss = recon + inter
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        tot_recon += recon.item()
        tot_inter += inter.item()
        # if batch_idx % 100 == 0:
        #     print(f"Epoch [{epoch}] Batch [{batch_idx}] | Total Loss: {loss.item():.6f} | Recon: {recon.item():.6f} | Inter: {inter.item():.6f}")

def run_standard_training():
    print(f"\n{'='*20} Starting Standard Training {'='*20}")
    setup_seed(seed)
    model = MyNet(dims, args.feature_dim, view, args.h_dims, args.fusion_indices, args.steps, args.alpha, args.temperature, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, device).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    for epoch in range(1, args.mse_epochs+1):
        train_one_epoch(model, optimizer, criterion, epoch, lambda_uniformity=args.lambda2, lambda_alignment=args.lambda1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_train_time = time.time() - start_time
    acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num)

    model_path = os.path.join("models", f"{args.dataset}.pth")
    torch.save(model.state_dict(), model_path)

    print(f"\n====== Standard Training Results ======")
    print(f"ACC: {acc:.4f} | NMI: {nmi:.4f} | PUR: {pur:.4f} | ARI: {ari:.4f} | Total Train Time: {total_train_time:.2f}s | Avg/Epoch: {total_train_time/args.mse_epochs:.4f}s")
    print(f"Model saved to: {model_path}")

if __name__ == '__main__':
    run_standard_training()
