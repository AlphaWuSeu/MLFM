from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import argparse
from read_ImageNetData import ImageNetData
import se_resnet
import se_resnext
import thop
import torch
import torch.nn as nn
import yacs.config
from src.dataset import Imagenet
from torch.utils.data import DataLoader


def count_op(model):
    data = torch.zeros((1, 3,
                        256, 256),

                       dtype=torch.float32,
                       device=torch.device('cuda'))
    model = model.cuda()
    data = data.cuda()
    return thop.clever_format(thop.profile(model, (data,), verbose=False))


def train_model(args, model, criterion, optimizer, scheduler, num_epochs, train_loader, test_loader):
    since = time.time()
    resumed = False

    best_acc = 0.0  # Keep track of the best accuracy so far

    best_model_wts = model.state_dict()

    for epoch in range(args.start_epoch + 1, num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if args.start_epoch > 0 and (not resumed):
                    scheduler.step(args.start_epoch + 1)
                    resumed = True
                else:
                    scheduler.step(epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            tic_batch = time.time()
            # Iterate over data.
            if phase == 'train':
                total_loss = 0.0
                total_corrects = 0
                dataset_sizes = len(train_loader.dataset)

                for i, (inputs, labels) in enumerate(train_loader):
                    # wrap them in Variable
                    running_loss = 0.0
                    running_corrects = 0
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    # running_loss += loss.data[0]
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)

                    total_loss += running_loss
                    total_corrects += running_corrects.item()

                    batch_loss = total_loss / ((i + 1) * args.batch_size)
                    batch_acc = total_corrects / ((i + 1) * args.batch_size)

                    if phase == 'train' and i % args.print_freq == 0:
                        print(
                            '[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                                epoch, num_epochs - 1, i, round(dataset_sizes / args.batch_size) - 1,
                                scheduler.get_lr()[0], phase, batch_loss, batch_acc, \
                                       args.print_freq / (time.time() - tic_batch)))
                        tic_batch = time.time()
                epoch_loss = total_loss / len(train_loader.dataset)
                epoch_acc = total_corrects / len(train_loader.dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            if phase == 'val':
                total_loss = 0.0
                total_corrects = 0
                dataset_sizes = len(test_loader.dataset)
                for i, (inputs, labels) in enumerate(test_loader):
                    running_loss = 0.0
                    running_corrects = 0
                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # statistics
                    # running_loss += loss.data[0]
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)

                    total_loss += running_loss
                    total_corrects += running_corrects.item()

                    batch_loss = total_loss / ((i + 1) * args.batch_size)
                    batch_acc = total_corrects / ((i + 1) * args.batch_size)

                    if phase == 'train' and i % args.print_freq == 0:
                        print(
                            '[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                                epoch, num_epochs - 1, i, round(dataset_sizes / args.batch_size) - 1,
                                scheduler.get_lr()[0], phase, batch_loss, batch_acc, \
                                       args.print_freq / (time.time() - tic_batch)))
                        tic_batch = time.time()
                epoch_loss = total_loss / len(test_loader.dataset)
                epoch_acc = total_corrects / len(test_loader.dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

        if (epoch + 1) % args.save_epoch_freq == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model, os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth.tar"))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str, default="/ImageNet")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-class', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.045)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output")
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--network', type=str, default="se_resnet_50", help="")
    args = parser.parse_args()

    args.data_dir = "/home/ac/datb/wfz_data/imagenet100"
    args.network = "se_resnet_10"
    # read data
    # dataloders, dataset_sizes = ImageNetData(args)
    training_params = {"batch_size": args.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "num_workers": 12}

    test_params = {"batch_size": args.batch_size//10,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": 12}
    training_set = Imagenet(root_dir=args.data_dir, mode="train")
    training_generator = DataLoader(training_set, **training_params)

    test_set = Imagenet(root_dir=args.data_dir, mode="val")
    test_generator = DataLoader(test_set, **test_params)

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    script_name = '_'.join([args.network.strip().split('_')[0], args.network.strip().split('_')[1]])

    if script_name == "se_resnet":
        model = getattr(se_resnet ,args.network)(num_classes = args.num_class)
    elif script_name == "se_resnext":
        model = getattr(se_resnext, args.network)(num_classes=args.num_class)
    else:
        raise Exception("Please give correct network name such as se_resnet_xx or se_rexnext_xx")


    macs, n_params = count_op(model)

    print("macs",macs)
    print("n_params",n_params)
    #sys.exit()
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
            model.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])
        model = torch.nn.DataParallel(model)
    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.98)

    model = train_model(args=args,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=args.num_epochs,
                           train_loader=training_generator,
                           test_loader=test_generator)
