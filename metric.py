from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from torch_clustering.kmeans.kmeans import PyTorchKMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def inference(loader, model, device, view, data_size):
    model.eval()
    Zs = [[] for _ in range(view)]
    combined_zs = []
    combined_xs = []
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, zs, _ = model.forward(xs)
            combined_x = torch.cat([xs[v] for v in range(view)], dim=1)
            combined_xs.append(combined_x)

            combined_z = torch.cat(zs, dim=1)
            combined_zs.append(combined_z) 

        for v in range(view):
            Zs[v].append(zs[v].detach())
        labels_vector.extend(y.cpu().numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    for v in range(view):
        Zs[v] = torch.cat(Zs[v], dim=0) 
    combined_xs = torch.cat(combined_xs, dim=0)
    combined_zs = torch.cat(combined_zs, dim=0)

    return labels_vector, Zs, combined_xs, combined_zs


def valid(model, device, dataset, view, data_size, class_num):
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    labels_vector, Zs, combined_xs, combined_zs = inference(test_loader, model, device, view, data_size)
    kmeans = PyTorchKMeans(n_clusters=class_num, n_init=100, verbose=False)
    print("Clustering results on combined_zs across all views (增强特征):")
    y_pred_gpu = kmeans.fit_predict(combined_zs)
    y_pred = y_pred_gpu.cpu().numpy()
    nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))

    # tSNE_PLOT(combined_xs.cpu().numpy(), labels_vector, name = f"{dataset.__class__.__name__}_raw_data")
    # tSNE_PLOT(combined_zs.cpu().numpy(), labels_vector, name = f"{dataset.__class__.__name__}")
    return acc, nmi, pur, ari

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(6, 6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    fixed_colors = [
        (0.894, 0.102, 0.110, 0.8),
        (0.216, 0.494, 0.722, 0.8),
        (0.302, 0.686, 0.290, 0.8),
        (0.596, 0.306, 0.639, 0.8),
        (1.000, 0.498, 0.000, 0.8),
        (1.000, 1.000, 0.200, 0.8),
        (0.651, 0.337, 0.157, 0.8),
        (0.969, 0.506, 0.749, 0.8),
        (0.600, 0.600, 0.600, 0.8),
        (0.300, 0.500, 0.400, 0.8)
    ]

    unique_labels = np.unique(label)
    for l in unique_labels:
        idx = np.where(label == l)[0]
        color = fixed_colors[l % len(fixed_colors)] 
        ax.scatter(
            data[idx, 0], data[idx, 1], 
            color=color, 
            s=50, 
            marker='o',
            edgecolors='none' 
        )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def tSNE_PLOT(Z, Y, name="xxx"):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    F = tsne.fit_transform(Z)
    fig = plot_embedding(F, Y, name)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    if not os.path.exists("images/png"):
        os.makedirs("images/png")
    if not os.path.exists("images/pdf"):
        os.makedirs("images/pdf")
    png_path = f"images/png/{name}.png"
    pdf_path = f"images/pdf/{name}.pdf"
    fig.savefig(png_path, format='png', transparent=True, dpi=500, pad_inches=0)
    fig.savefig(pdf_path, format='pdf', transparent=True, dpi=500, pad_inches=0)
    print(f"t-SNE plot saved as {png_path} and {pdf_path}")
    plt.close(fig)
