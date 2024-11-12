# https://jimmy-ai.tistory.com/342
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
import os
import numpy as np
import torch
import torch.nn as nn
import time

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# # https://github.com/pytorch/pytorch/issues/21987
# def nanmean(v, *args, inplace=False, **kwargs):
#     if not inplace:
#         v = v.clone()
#     is_nan = torch.isnan(v)
#     v[is_nan] = 0
#     return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

# def seq2list_cuda(seq,device):
#     nan_value = -99999
#     ret_seq = []
#     k=0
#     N=len(seq)
#     for x in seq:
#         start_seq = torch.tensor([nan_value]*k).to(device).float()
#         end_seq   = torch.tensor([nan_value]*(N-k-1)).to(device).float()
#         x = x.to(device)
        
#         if len(start_seq)==0:
#             _seq  = torch.cat([x,end_seq],axis=0)
#         elif len(end_seq)==0:
#             _seq  = torch.cat([start_seq,x],axis=0)
#         else:
#             _seq  = torch.cat([start_seq,x,end_seq],axis=0)
            
#         _seq[_seq==nan_value] = float('nan')
#         ret_seq.append(_seq)
#         k+=1
#     ret_seq = torch.stack(ret_seq,dim=0)
#     #print('(1)',ret_seq)
#     ret_seq = nanmean(ret_seq,dim=0)
#     #print('(2)',ret_seq)
#     return ret_seq
        
def train(
    model, train_loader, val_loader, epochs,
    optimizer, criterion, scheduler=None, 
    early_stopping=False, early_stopping_patience=10, early_stopping_verbose=False,
    device='cpu', metric_period=1, 
    verbose=True, save_model_path = './mc/best_model.pt',
    custom_metric=None,
):
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    assert isinstance(early_stopping,bool), "early_stopping must by type bool"
    
    es = EarlyStopping(
        patience=early_stopping_patience,
        verbose=early_stopping_verbose,
        path=save_model_path
    )
    
    model.to(device)

    best_loss  = float('inf')
    best_loss_org = float('inf')
    best_epoch = 1
    best_model = None
    is_best    = None
    
    start_time = time.time()
    epoch_s = time.time()
    for epoch in range(1, epochs+1):
        
        model.train()
        train_loss = []
        train_custom = []
        for X, Y in iter(train_loader):

            X = X.float().to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            output = model(X)

            loss = criterion(output, Y)
            loss.backward()  # Getting gradients
            optimizer.step() # Updating parameters

            train_loss.append(loss.item())
            if custom_metric is not None:
                custom = custom_metric(output, Y)
                train_custom.append(custom.item())
            
        val_loss, val_custom = validation(model, val_loader, criterion, device, custom_metric)

        epoch_e = time.time()
            
        if scheduler is not None:
            scheduler.step(val_loss)

        # update the best epoch & best loss
        if (best_loss > val_loss) | (epoch==1):
            best_epoch = epoch
            best_loss = val_loss
            best_custom = val_custom
            best_model = model
            is_best = 1
            torch.save(best_model.state_dict(), save_model_path)
        else:
            is_best = 0
            
        # 결과물 printing
        if ((verbose) & (epoch % metric_period == 0)) | (epoch==epochs):
            mark = '*' if is_best else ' '
            epoch_str = str(epoch).zfill(len(str(epochs)))
            
            if custom_metric is None:
                progress = '{}[{}/{}] loss: {:.4f}, val_loss: {:.4f}, best: {:.4f}({}) | elapsed: {:.1f}s, total: {:.1f}s, remaining: {:.1f}s'\
                    .format(
                        mark, epoch_str, epochs,
                        np.mean(train_loss), val_loss, best_loss, best_epoch,
                        epoch_e-epoch_s, epoch_e-start_time, (epoch_e-epoch_s)*(epochs-epoch)/metric_period,
                    )
            else:
                progress = '{}[{}/{}] loss: {:.4f}, val_loss: {:.4f}, best: {:.4f}({}) | custom: {:.4f}, val_custom: {:.4f}, best: {:.4f} | elapsed: {:.1f}s, total: {:.1f}s, remaining: {:.1f}s'\
                    .format(
                        mark, epoch_str, epochs,
                        np.mean(train_loss), val_loss, best_loss, best_epoch,
                        np.mean(train_custom), val_custom, best_custom,
                        epoch_e-epoch_s, epoch_e-start_time, (epoch_e-epoch_s)*(epochs-epoch)/metric_period,
                    )
            epoch_s = time.time()
            print(progress)

        # early stopping 여부를 체크. 현재 과적합 상황 추적
        if early_stopping:
            es(val_loss, model)
            if es.early_stop:
                mark = '*' if is_best else ' '
                epoch_str = str(epoch).zfill(len(str(epochs)))
                
                if custom_metric is None:
                    progress = '{}[{}/{}] loss: {:.4f}, val_loss: {:.4f}, best: {:.4f}({}) | elapsed: {:.1f}s, total: {:.1f}s, remaining: {:.1f}s'\
                        .format(
                            mark, epoch_str, epochs,
                            np.mean(train_loss), val_loss, best_loss, best_epoch,
                            epoch_e-epoch_s, epoch_e-start_time, (epoch_e-epoch_s)*(epochs-epoch)/metric_period,
                        )
                else:
                    progress = '{}[{}/{}] loss: {:.4f}, val_loss: {:.4f}, best: {:.4f}({}) | custom: {:.4f}, val_custom: {:.4f}, best: {:.4f} | elapsed: {:.1f}s, total: {:.1f}s, remaining: {:.1f}s'\
                        .format(
                            mark, epoch_str, epochs,
                            np.mean(train_loss), val_loss, best_loss, best_epoch,
                            np.mean(train_custom), val_custom, best_custom,
                            epoch_e-epoch_s, epoch_e-start_time, (epoch_e-epoch_s)*(epochs-epoch)/metric_period,
                        )
                epoch_s = time.time()
                print('< Early Stopping Activated >')
                print(progress)
                break

    return best_model

def validation(model, val_loader, criterion, device, custom_metric=None):
    model.eval()
    val_loss = []
    val_custom = []
    with torch.no_grad():
        for X, Y in iter(val_loader):
            X = X.float().to(device)
            Y = Y.to(device)
            
            output = model(X)
            loss = criterion(output, Y)
            val_loss.append(loss.item())

            if custom_metric is not None:
                custom = custom_metric(output, Y)
                val_custom.append(custom.item())

    if custom_metric is not None:
        return np.mean(val_loss), np.mean(val_custom)
    else:
        return np.mean(val_loss), np.mean(val_loss)

def predict(model, loader, device, inverse_transform=None):
    model.to(device)
    model.eval()
    
    true_list = []
    pred_list = []
    with torch.no_grad():
        for X, Y in iter(loader):
            X = X.float().to(device)

            output = model(X)
            if inverse_transform is not None:
                output = inverse_transform(output)
            output = output.cpu().numpy().tolist()

            if inverse_transform is not None:
                Y  = inverse_transform(Y)
            Y = Y.cpu().numpy().tolist()

            true_list += Y
            pred_list += output

    return true_list, pred_list

def inference(model, loader, device, inverse_transform=None):
    model.to(device)
    model.eval()
    
    true_list = []
    pred_list = []
    with torch.no_grad():
        for X in iter(loader):
            X = X.float().to(device)

            output = model(X)
            if inverse_transform is not None:
                output = inverse_transform(output)
            output = output.cpu().numpy().tolist()

            pred_list += output

    return pred_list


# #-------------------------------------------------------------------------------------------#
# # Example
# #-------------------------------------------------------------------------------------------#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader, TensorDataset

# print('> torch version  :',torch.__version__)
# print('> cuda available :',torch.cuda.is_available())

# class CustomDataset(Dataset):
#     def __init__(self,X,y,infer_mode):
#         self.infer_mode = infer_mode
        
#         window_size = 1
#         sequence_length = 180 #N_PREDICT
        
#         X = X[X.shape[0] % sequence_length:]

#         self.X_list = []
#         self.y_list = []
#         for i in range(len(X)):
#             seq_x = X.iloc[i].values
#             seq_y = y.iloc[i].values
#             self.X_list.append(torch.Tensor(seq_x))
#             self.y_list.append(torch.Tensor(seq_y))

#     def __getitem__(self, index):
#         data  = self.X_list[index]
#         label = self.y_list[index]
#         if self.infer_mode == False:
#             return data, label
#         else:
#             return data

#     def __len__(self):
#         return len(self.X_list)
    
# batch_size  = 16
# num_workers = 8

# train_dataset = CustomDataset(X=X_train, y=y_train, infer_mode=False)
# train_loader  = DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=num_workers)

# valid_dataset = CustomDataset(X=X_valid, y=y_valid, infer_mode=False)
# valid_loader  = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False, num_workers=num_workers)

# model = CustomModel()

# model.eval()
# optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-3, weight_decay=1e-5)
# # optimizer = torch.optim.SGD(params = model.parameters(), lr = 1e-2, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs',min_lr=1e-7, verbose=True)

# best_model = train(
#     model,
#     optimizer=optimizer,
#     train_loader=train_loader,
#     valid_loader=valid_loader,
#     scheduler=scheduler,
#     device=device,
#     early_stopping=True,
#     early_stopping_patience=10,
#     early_stopping_verbose=False,
#     metric_period=1,
#     epochs=256,
#     transform_y='identity',
#     verbose=True,
#     print_shape=False,
#     save_model_path = './mc/best_model.pt',
# )