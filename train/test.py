import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

def test(args, model, features, adj, labels, idx_test, class_weights):
    model.eval()
    output, _, _, _, _ = model(features, adj)

    loss = torch.nn.CrossEntropyLoss(weight = class_weights).to(output.device)
    loss_test = loss(output[idx_test], labels[idx_test])

    scores = F.softmax(output, dim=1)
    fpr, tpr, t = roc_curve(labels[idx_test].detach().cpu().numpy(), scores[idx_test,0].detach().cpu().numpy(), pos_label = 0)
    roc_auc= auc(fpr, tpr)    
    return roc_auc
