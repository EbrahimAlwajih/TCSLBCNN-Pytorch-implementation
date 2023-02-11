import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.utils.data
from tqdm import tqdm
import torchvision
from tcslbcnn_model import TCSLBCNN
from utils import calc_accuracy, get_mnist_loader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("runs/cifar10")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'tcslbcnn_best.pt')
torch.cuda.empty_cache()
import sys

def test(model=None):
    if model is None:
        assert os.path.exists(MODEL_PATH), "Train a model first"
        tcslbcnn_depth, state_dict = torch.load(MODEL_PATH)
        model = TCSLBCNN(depth=tcslbcnn_depth)
        model.load_state_dict(state_dict)
    loader = get_mnist_loader(train=False)
    accuracy = calc_accuracy(model, loader=loader, verbose=True)
    print("MNIST test accuracy: {:.3f}".format(accuracy))


def train(n_epochs=50,nInputPlane=3, tcslbcnn_depth=2, batch_size=256, learning_rate=1e-2, momentum=0.9, weight_decay=1e-4, lr_scheduler_step=5):
    start = time.time()
    models_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    if nInputPlane==3:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize([28, 28]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif nInputPlane==1:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize([28, 28]),
            transforms.Normalize((0.5), (0.5))])
   
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False)

    #classes = ('plane', 'car', 'bird', 'cat',
           #    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    #train_loader = get_mnist_loader(train=True)
    #test_loader = get_mnist_loader(train=False)
        
    model = TCSLBCNN(depth=tcslbcnn_depth,nInputPlane=nInputPlane)
    ############## TENSORBOARD ########################
    examples = iter(test_loader) 
    example_data, example_targets = examples.next()
    writer.add_graph(model, example_data)
    #writer.close()
    #sys.exit()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    best_accuracy = 0.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate,
                      weight_decay=weight_decay)
    # toch.optim.RAdam(params,lr=0.005,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)
    #optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate,
    #                  momentum=momentum, weight_decay=weight_decay, nesterov=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step)
    
    for epoch in range(n_epochs):
        for batch_id, (inputs, labels) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, n_epochs))):
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        accuracy_train = calc_accuracy(model, loader=train_loader)
        accuracy_test = calc_accuracy(model, loader=test_loader)
        print("Epoch {} accuracy: train={:.3f}, test={:.4f}".format(epoch, accuracy_train, accuracy_test))
        if accuracy_test > best_accuracy:
            best_accuracy = accuracy_test
            torch.save((tcslbcnn_depth, model.state_dict()), MODEL_PATH)
        scheduler.step(epoch=epoch)
        # print(model.chained_blocks[0].conv_tcslbp.weight)
        # print(model.chained_blocks[0].conv_1x1.weight)
    train_duration_sec = int(time.time() - start)
    print('Finished Training. Total training time: {} sec'.format(train_duration_sec))
    

if __name__ == '__main__':
    # train includes test phase at each epoch 
    # nInputPlane is the number of channel of the images, nInputPlane for MNIST is 1 , CIFAR10 is 3, etc.
    train(n_epochs=80,nInputPlane=3, tcslbcnn_depth=15,batch_size=16, learning_rate=1e-3, momentum=0.9, weight_decay=1e-4, lr_scheduler_step=30)
