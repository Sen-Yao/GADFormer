import argparse
import torch
import numpy as np
import torch.optim as optim
import time
from tqdm import tqdm

from utils import load_data
from models import GCN

from train.train import train
from train.test import test


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=5,
                    help='class-balanced parameter.')
parser.add_argument('--beta', type=float, default=1e-1,
                    help='l2 regularization parameter).')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_test, idx_val = load_data("cora")  #cora, citeseer, pubmed

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
weights=[args.alpha, 1]
class_weights = torch.FloatTensor(weights)


# Train model
t_total = time.time()
max_auc = 0
with tqdm(range(args.epochs), desc="Training") as pbar:
    for epoch in pbar:
        loss_train, loss_val, epoch_time = train(args, model, optimizer, features, adj, labels, idx_train, idx_val, class_weights)
        roc_auc = test(args, model, features, adj, labels, idx_test, class_weights)
        if roc_auc > max_auc:
            max_auc = roc_auc
        pbar.set_postfix({
            "loss_train": f"{loss_train:.4f}",
            "loss_val": f"{loss_val:.4f}",
            "time": f"{epoch_time:.4f}s"
        })
AUC = max_auc
print('AUC: {:04f}'.format(AUC))
