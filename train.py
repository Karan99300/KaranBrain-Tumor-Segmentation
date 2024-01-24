import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

def train(model,loader,optimizer,loss_fn,device):
    model.train()
    
    train_loss,train_dice=0,0
    
    for batch, (X,y) in enumerate(loader):
        X,y=X.to(device),y.to(device)
        
        y_pred=model(X)
        loss=loss_fn(y_pred,y)
        train_loss+=loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predicted_class=torch.sigmoid(y_pred)
        predicted_class=(predicted_class>0.5).float()
        
        eps=1e-8
        train_dice+=(
            (2*(y*predicted_class).sum()+eps)/((y+predicted_class).sum()+eps)
            ).cpu().item()
        
    train_loss=train_loss/len(loader)
    train_dice=train_dice/len(loader)
    
    return train_loss, train_dice

def eval(model,loader,loss_fn,device):
    model.eval()
    
    eval_loss,eval_dice=0,0
    
    with torch.inference_mode():
        for batch, (X,y) in enumerate(loader):
            X,y=X.to(device),y.to(device)
            
            y_pred=model(X)
            loss=loss_fn(y_pred,y)
            eval_loss+=loss.item()
            
            predicted_class=torch.sigmoid(y_pred)
            predicted_class=(predicted_class>0.5).float()
            
            eps=1e-8
            eval_dice+=(
                (2*(y*predicted_class).sum()+eps)/((y+predicted_class).sum()+eps)
                ).cpu().item()
            
        eval_loss=eval_loss/len(loader)
        eval_dice=eval_dice/len(loader)
    
    return eval_loss,eval_dice

def epochs(model, train_loader, val_loader, optimizer, scheduler, loss_fn, epochs, device,save_path):
    session = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }

    start_epoch = 0  
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded model checkpoint from '{save_path}'")

        # Extract the epoch number from the save_path
        start_epoch = int(save_path.split('_')[-2]) + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, start_epoch + epochs)):
        print(f'\nEpoch {epoch + 1}/{start_epoch + epochs}')
        train_loss, train_dice = train(model, train_loader, optimizer, loss_fn, device)
        eval_loss, eval_dice = eval(model, val_loader, loss_fn, device)

        # Save model state_dict
        torch.save(model.state_dict(), f'Unet_Model_epoch_{epoch}.pth')

        current_lr = 0
        if scheduler:
            scheduler.step(eval_loss)
            current_lr = optimizer.param_groups[0]['lr']

        log_text = f'loss: {train_loss:.4f} - dice_score: {train_dice:.4f} - eval_loss: {eval_loss:.4f} - eval_dice_score: {eval_dice:.4f}'

        if scheduler:
            print(log_text + f' - lr: {current_lr}')
        else:
            print(log_text)

        session['train_loss'].append(train_loss)
        session['train_dice'].append(train_dice)
        session['val_loss'].append(eval_loss)
        session['val_dice'].append(eval_dice)

    return session

def predict(model, loader, device, threshold=0.5):
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(tqdm(loader)):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            predicted_class = torch.sigmoid(y_pred)
            predicted_class = (predicted_class >= threshold).float()
            predictions.append(predicted_class.cpu().numpy())

    return np.vstack(predictions)
    
        
        