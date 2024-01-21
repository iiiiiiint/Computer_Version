import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


class Data:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 归一化
        ])
        # 读取数据
        train_data = datasets.MNIST(root='./data', train=True, transform=transform_test, download=False)
        test_data = datasets.MNIST(root='./data', train=False, transform=transform_test, download=False)
        # 建立dataloader
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader


# 卷积神经网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.max_acc = 0.0
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(batch_size, 1,28,28) output(batch_size, 16,24,24)
        x = self.pool1(x)  # output(batch_size, 16，12，12)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))  # output(batch_size, 32,10.10)
        x = self.pool2(x)  # output(batch_size, 32,5,5)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))  # output(batch_size, 32,3,3)
        x = self.pool3(x)  # output(batch_size, 32,2,2)
        x = self.bn3(x)
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x)) # output(batch_size, 120)
        x = F.relu(self.fc2(x)) # output(batch_size, 84)
        x = self.fc3(x)  # output(batch_size, 10)
        x = F.log_softmax(x, dim=1)
        return x


def train(model, train_dataloader, device, dataset):
    model.train()  # 调成训练模式
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Training...")
    running_loss = 0.0
    stop = 0
    pre_acc = 0.0
    for epoch in range(10):
        for step, batch_data in enumerate(train_dataloader):
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            out = model(x)
            # print(out.shape)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if step % 200 == 0:
            #     print("epoch={}, step={}, loss={:5f}".format(epoch, step, float(running_loss/200)))
            #     running_loss = 0.0

        cur_acc = predict(model, dataset.get_test_loader(), device, epoch)
        # 如果连续5次效果没有明显增长，则立刻终止训练
        if cur_acc < pre_acc or cur_acc - pre_acc < 0.0001 or cur_acc < model.max_acc:
            stop += 1
        else:
            stop = 0
        pre_acc = cur_acc

        if stop > 5:
            print("Stop Early...")
            break


def predict(model, test_loader, device, epoch):
    model.to(device)
    model.eval()
    correct, total = 0.0, 0.0
    with torch.no_grad():
        for step, batch_data in enumerate(test_loader):
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            # print(x.shape)
            # exit()
            pred = outputs.max(1, keepdim=True)[1]
            # print(pred.shape)
            # exit()
            total = total + y.size(0)
            correct = correct + pred.eq(y.view_as(pred)).sum().item()  # 计算准确率
    cur_acc = correct / total
    print('epoch:{} Accuracy:{:.4f}%'.format(epoch, 100.0 * correct / total))
    if cur_acc > model.max_acc:
        model.max_acc = cur_acc
        # print("Max_Acc:{}".format(model.max_acc))
        torch.save(model, "./model/LeNet.pt")  # 保存表现最好的模型
    return cur_acc


def main_worker():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_obj = LeNet()
    dataset = Data(32)  # batch_size 为 32
    train(model_obj, dataset.get_train_loader(), device, dataset)
    print("Training done...")
    print("The max accuracy is {}%".format(model_obj.max_acc * 100))


if __name__ == '__main__':
    main_worker()
