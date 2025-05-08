import time
import torch
import torch.nn.functional as F


def train(args, model, optimizer, features, adj, labels, idx_train, idx_val, class_weights):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, w1, w2, w3, w4 = model(features, adj)
    
    w1 = torch.pow(torch.norm(w1), 2)
    w2 = torch.pow(torch.norm(w2), 2)
    w3 = torch.pow(torch.norm(w3), 2)
    w4 = torch.pow(torch.norm(w4), 2)
    l2_reg = w1 + w2 + w3 + w4 

    loss = torch.nn.CrossEntropyLoss(weight=class_weights).to(output.device)
    loss_train = loss(output[idx_train], labels[idx_train]) + args.beta*l2_reg
    loss_train.backward()
    optimizer.step()
    
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output, w1, w2, w3, w4 = model(features, adj) 
        
    loss_val = loss(output[idx_val], labels[idx_val]) + args.beta*l2_reg
    return loss_train.item(), loss_val.item(), time.time() - t