import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Utils.getData import getImageLabel
from torch.optim import Adam
from torchvision import models
from Model.cnn import SimpleCNN

def main():
    BATCH_SIZE = 32
    EPOCH = 10
    LEARNING_RATE = 0.001
    folds = [1,2,3,4,5]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_aug_loader = DataLoader(getImageLabel(augmented=f'd:/Kuliah/Semester 3/IPSD/Deep Learning/Dataset/Augmented Images/Augmented Images/FOLDS_AUG/', folds=folds, subdir=['Train']), batch_size=BATCH_SIZE, shuffle=True)
    train_ori_loader = DataLoader(getImageLabel(original=f'd:/Kuliah/Semester 3/IPSD/Deep Learning/Dataset/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Train']), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(getImageLabel(original=f'd:/Kuliah/Semester 3/IPSD/Deep Learning/Dataset/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Valid']), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(input_dim=32, input_c=3, output=6, dropout=0.5, device=DEVICE)
    model.to(DEVICE)
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    loss_train_all, loss_valid_all = [], []
    for epoch in range(EPOCH):
        train_loss = 0
        valid_loss = 0
        model.train()
        for batch, (src, trg) in enumerate(train_aug_loader):
            src = src.permute(0, 3, 1, 2).to(DEVICE)
            trg = trg.to(DEVICE)
            
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        for batch, (src, trg) in enumerate(train_ori_loader):
            src = src.permute(0, 3, 1, 2).to(DEVICE)
            trg = trg.to(DEVICE)
                         
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        for batch, (src, trg) in enumerate(valid_loader):
            src = src.permute(0, 3, 1, 2).to(DEVICE)
            trg = trg.to(DEVICE)

            pred = model(src)
            loss = loss_function(pred, trg)
            valid_loss += loss.detach().numpy()
            
        loss_train_all.append(train_loss / (len(train_aug_loader) + len(train_ori_loader)))
        loss_valid_all.append(valid_loss / len(valid_loader))
        print("epoch = ", epoch + 1, ", train loss = ", train_loss / (len(train_aug_loader) + len(train_ori_loader)),
            ", validation loss = ", valid_loss / len(valid_loader))
            
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / (len(train_aug_loader) + len(train_ori_loader)),
            }, "./SimpleCNN_" + str(epoch + 1) + ".pt")
            
    plt.plot(range(EPOCH), loss_train_all, color="#931a00", label='Training')
    plt.plot(range(EPOCH), loss_valid_all, color="#3399e6", label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./SimpleCNN.png")

if __name__ == "__main__":
    main()