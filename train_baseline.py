import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import torch.cuda.amp
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, get_training_dataloader_CIFAR, get_test_dataloader_CIFAR
import ipdb
from down_sample.down_sample_pycls import downsample_load_resnet


def train(epoch, args):
    settings.IS_TRAIN = True
    running_loss = 0
    start = time.time()
    net.train()
    correct = 0.0
    num_sample = 0
    for batch_index, (images, labels) in enumerate(CIFAR_training_loader):
        if args.gpu:
            labels = labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
        num_sample += images.size()[0]
        optimizer.zero_grad()
        with autocast():
            outputs = net(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n_iter = (epoch - 1) * len(CIFAR_training_loader) + batch_index + 1
        if batch_index % 10 == 9:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                running_loss/10,
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(CIFAR_training_loader.dataset)
            ))
            print('training time consumed: {:.2f}s'.format(time.time() - start))
            if args.local_rank == 0:
                writer.add_scalar('Train/avg_loss', running_loss/10, n_iter)
                writer.add_scalar('Train/avg_loss_numpic', running_loss/10, n_iter * args.b)
            loss_temp = running_loss / 10
            running_loss = 0
    finish = time.time()
    if args.local_rank == 0:
        writer.add_scalar('Train/acc', correct/num_sample*100, epoch)
    print("Training accuracy: {:.2f} of epoch {}".format(correct/num_sample*100, epoch))
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return loss_temp


@torch.no_grad()
def eval_training(epoch, args, result_path):
    settings.IS_TRAIN = False
    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0
    real_batch = 0
    for (images, labels) in CIFAR_test_loader:
        real_batch += images.size()[0]
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}%, Time consumed:{:.2f}s'.format(
            test_loss * args.b / len(CIFAR_test_loader.dataset),
            correct.float() / real_batch * 100,
            finish - start
        ))


    if args.local_rank == 0:
        # add information to tensorboard
        writer.add_scalar('Test/Average loss', test_loss * args.b / len(CIFAR_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / real_batch * 100, epoch)

    return correct.float() / real_batch * 100, correct.float() / len(CIFAR_test_loader.dataset), test_loss * args.b / len(CIFAR_test_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-dataset', type=str, default='cifar100', help='dataset name')
    parser.add_argument('-b', type=int, default=100, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()
    print(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    SEED = 445
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    net = get_network(args)
    net.cuda()
    
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
    num_gpus = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # data preprocessing:
    CIFAR_training_loader = get_training_dataloader_CIFAR(sampler=1, batch_size=args.b//num_gpus, num_workers=4, shuffle=False, dataset=args.dataset)

    CIFAR_test_loader = get_test_dataloader_CIFAR(sampler=1, batch_size=args.b//num_gpus, num_workers=4, shuffle=False, dataset=args.dataset)

    # learning rate should go with batch size.
    b_lr = args.lr
    loss_function = nn.CrossEntropyLoss(reduction='mean')
    # optimizer = optim.SGD([{'params': net.parameters(), 'initial_lr': b_lr}], momentum=0.9, lr=b_lr, weight_decay=0.0001) # cifar100
    optimizer = optim.SGD([{'params': net.parameters(), 'initial_lr': b_lr}], momentum=0.9, lr=b_lr, weight_decay=0.0001) # cifar10

    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1) # cifar100
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1) # cifar10
    iter_per_epoch = len(CIFAR_training_loader)
    LOG_INFO = "Pretrain-stage"
    checkpoint_path = os.path.join('checkpoint_step_experiment_'+args.dataset, args.net, 'from_zero', str(args.b), str(args.lr), settings.TIME_NOW)
    result_path = os.path.join("result", 'step_experiment', 'from_zero', str(args.b), str(args.lr), settings.TIME_NOW)
    if args.local_rank == 0:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    
    '''
    # # to load a pretrained model
    # Path = "checkpoint/WSresnet18/512/1.6/Pretrain-stage/Tuesday_28_February_2023_18h_15m_06s/WSresnet18-199-regular.pth"  # change to your ckpt
    Path = "checkpoint/WSresnet18/4GPU/downsample_34_18/128/0.4/Pretrain-stage/Monday_13_March_2023_15h_02m_40s/WSresnet18-100-regular.pth"  # change to your ckpt
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    checkpoint = torch.load(Path, map_location=map_location)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for groups in optimizer.param_groups:
        groups['lr'] = b_lr
    '''
    # use tensorboard
    if args.local_rank == 0:
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, str(args.b), str(args.lr), LOG_INFO, settings.TIME_NOW))

    # create checkpoint folder to save model
    if args.local_rank == 0:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    settings.EPOCH = 200
    for epoch in range(1, settings.EPOCH+1):
        settings.EPOCH_INDEX = epoch
        print(settings.EPOCH_INDEX)
        train_loss = train(epoch, args)

        train_scheduler.step()
        acc, _ , eval_loss= eval_training(epoch, args, result_path)
        # Computational average accuracy
        accuracy_tensor = torch.tensor(acc.clone()).cuda()
        torch.distributed.all_reduce(accuracy_tensor, op=torch.distributed.ReduceOp.SUM)
        accuracy_mean = accuracy_tensor.item() / torch.distributed.get_world_size()
        if(args.local_rank == 0):
            with open(result_path+'/result.txt', 'a') as file0:
                print('Epoch: {:.0f}, Accuracy: {:.4f}%, Train_loss: {:.4f}, Eval_loss: {:.4f}\n'.format(
                    epoch,
                    accuracy_mean,
                    train_loss,
                    eval_loss
                ), file = file0)
        if best_acc < accuracy_mean and args.local_rank == 0:
            with open(result_path+'/result.txt', 'a') as file0:
                print('The best Epoch: {:.0f}, The best Accuracy: {:.4f}%, Train_loss: {:.4f}, Eval_loss: {:.4f}\n'.format(
                    epoch,
                    accuracy_mean,
                    train_loss,
                    eval_loss
                ), file = file0)
            best_acc = accuracy_mean
        if epoch > (settings.EPOCH-20) and best_acc < accuracy_mean and args.local_rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = accuracy_mean
            continue
        elif ((not epoch % settings.SAVE_EPOCH) and args.local_rank == 0):
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
            continue

    if args.local_rank == 0:
        writer.close()

