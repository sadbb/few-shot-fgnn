import argparse
from prettyprinter import pprint
import torch 
import numpy as np
import random
import os
from tools import *
from data import *
from torch.utils.data import DataLoader
from models.FGNN import FGNN
from torch import nn
from tensorboardX import SummaryWriter
import tqdm

    
class FGNN_trainer(object):
    def __init__(self, args):
        # log_dir
        log_dir = './logs/'
        ensure_path(log_dir)
        save_path = args.dataset + '_shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) + \
            '_gamma' + str(args.gamma) + '_lr1' + str(args.lr1) + '_lr2' + str(args.lr2) + \
            '_batch' + str(args.batch_size) + '_maxepoch' + str(args.max_epoch) + \
            '_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step) + \
            '_stepsize' + str(args.reduce_step) + '_' + args.exp_name
        args.save_path = log_dir + '/'  + save_path
        ensure_path(args.save_path)
        self.args = args

        self.trainset = Dataset(self.args.dataset_dir)
        self.train_sampler = Categories_Sampler(self.trainset.label, self.args.batch_size, self.args.way, self.args.shot + self.args.train_query)
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=4, pin_memory=True)

        self.valset = Dataset(self.args.dataset_dir, 'val')
        self.val_sampler = Categories_Sampler(self.valset.label, 600, self.args.way, self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=4, pin_memory=True)
        
        self.model = FGNN(self.args)
        # Optimizer 
        self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())},
            {'params': self.model.GNN_classifier.parameters(), 'lr': self.args.lr2},
            {'params': self.model.relation_encoder.parameters(), 'lr': self.args.lr2*10},], lr=self.args.lr1, weight_decay=1e-5)
        # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.reduce_step, gamma=self.args.gamma)     
        
        # load pretrained model without temporary classifier
        self.model_dict = self.model.state_dict()
        if self.args.init_weights is not None:
            self.load_fea_extractor()
        else:
            raise ValueError('No init_weights!')  

        self.model = self.model.cuda()
        if self.args.num_gpus > 1:
            print('Construct multi-gpu model ...')
            self.model = nn.DataParallel(self.model, device_ids=list(range(args.num_gpus)), dim=0)
            print('done!\n')

    def train(self):
        # Set the train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)
        mask = 1-torch.eye(5*(self.args.shot+self.args.train_query)).cuda()

        # Generate the labels for train set of the episodes
        label_shot = torch.arange(self.args.way).repeat(self.args.shot).type(torch.cuda.LongTensor)
        #label_edge = 1-label2edge(label_shot)
        # Start train
        for epoch in range(1, self.args.max_epoch + 1):
            # Update learning rate
            #print('=============epcoh===============')
            self.lr_scheduler.step()
            self.model.train()
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Generate the labels for test set of the episodes during train updates
            label = torch.arange(self.args.way).repeat(self.args.train_query+self.args.shot).type(torch.cuda.LongTensor)
            label_edge = 1-label2edge(label)

            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                data, _ = [_.cuda() for _ in batch]

                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
        
                logits, edge = self.model((data_shot, label_shot, data_query, label_edge))
        
                loss_e = F.binary_cross_entropy(edge, label_edge, reduce=False).cuda()
                # Calculate the loss of edge prediction
                loss_p = (loss_e *(1-label_edge)*mask).sum()/((1-label_edge)*mask).sum()
                loss_n = (loss_e*label_edge).sum()/label_edge.sum()
                loss_e = loss_p+loss_n
                # Calculate the loss of node classification
                loss = F.cross_entropy(logits, label[5*self.args.shot:])

                # Calculate the accuracies of node classification and edge prediction
                re_label = torch.eq((edge>0.5).float(), label_edge).float()
                num_y_p = (re_label*(1-label_edge)*mask).sum().item()
                num_p = ((1-label_edge)*mask).sum().item()
                num_y_n = (re_label*label_edge).sum().item()
                num_n = label_edge.sum().item()
                acce_p = ((re_label*(1-label_edge)*mask).sum()/((1-label_edge)*mask).sum()).item()
                acce_n = ((re_label*label_edge).sum()/label_edge.sum()).item()
                acce = ((re_label*mask).sum()/mask.sum()).item()
                acc = count_acc(logits, label[5*self.args.shot:])

                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)                
                writer.add_scalar('data/loss_e', float(loss_e), global_count)
                writer.add_scalar('data/acce', float(acce), global_count)
                # Print losses and accuracies for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f} Loss_e={:.4f} Loss_p={:.4f} Loss_n={:.4f} [{}/{}, Accp={:.4f}] \[{}/{}, Accn={:.4f}] \
                    Acce={:.4f}'.format(epoch, loss.item(), acc, loss_e.item(), loss_p.item(),  loss_n.item(), num_y_p, num_p, acce_p, num_y_n, num_n, acce_n,acce))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss_all = loss +0.5* loss_e
                loss_all.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # Start validation for this epoch, set model to eval mode
            self.model.eval()

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()
            val_losse_averager = Averager()
            val_acce_averager = Averager()

            # Generate the labels for test set of the episodes during val for this epoch
            label = torch.arange(self.args.way).repeat(self.args.val_query+self.args.shot).type(torch.cuda.LongTensor)
            label_edge = 1-label2edge(label)
            # Print previous information
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run validation
            tqdm_val = tqdm.tqdm(self.val_loader)
            for i, batch in enumerate(tqdm_val, 1):
                data, _ = [_.cuda() for _ in batch]

                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                logits, edge = self.model((data_shot, label_shot, data_query, label_edge))

                # Calculate the loss of edge prediction
                loss_e = F.binary_cross_entropy(edge, label_edge, reduce=False).cuda()
                loss_p = (loss_e *(1-label_edge)*mask).sum()/((1-label_edge)*mask).sum()
                loss_n = (loss_e*label_edge).sum()/label_edge.sum()
                loss_e = loss_p+loss_n

                # Calculate the loss of node classification
                loss = F.cross_entropy(logits, label[p:])

                # Calculate the accuracies of node classification and edge prediction
                re_label = torch.eq((edge>0.5).float(), label_edge).float()
                num_y_p = (re_label*(1-label_edge)*mask).sum().item()
                num_p = ((1-label_edge)*mask).sum().item()
                num_y_n = (re_label*label_edge).sum().item()
                num_n = label_edge.sum().item()
                acce_p = ((re_label*(1-label_edge)*mask).sum()/((1-label_edge)*mask).sum()).item()
                acce_n = ((re_label*label_edge).sum()/label_edge.sum()).item()
                acce = ((re_label*mask).sum()/mask.sum()).item()
                acc = count_acc(logits, label[p:])
                # Print loss and accuracy for this step
                tqdm_val.set_description('Epoch {}, Loss={:.4f} Acc={:.4f} Loss_e={:.4f} Loss_p={:.4f} Loss_n={:.4f} [{}/{}, Accp={:.4f}] [{}/{}, Accn={:.4f}] \
                    Acce={:.4f}'.format(epoch, loss.item(), acc, loss_e.item(), loss_p.item(),  loss_n.item(), num_y_p, num_p, acce_p, num_y_n, num_n, acce_n,acce))


                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)
                val_losse_averager.add(loss_e.item())
                val_acce_averager.add(acce)

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            val_losse_averager = val_losse_averager.item()
            val_acce_averager = val_acce_averager.item()
            #tval_acc_averager = tval_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)       
            writer.add_scalar('data/val_losse', float(val_losse_averager), epoch)
            writer.add_scalar('data/val_acce', float(val_acce_averager), epoch) 
            # Print loss and accuracy for this epoch
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f} Losse={:.4f} Acce={:.4f}'.format(epoch, val_loss_averager, val_acc_averager, val_losse_averager, val_acce_averager))

            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 10 == 0:
               self.save_model('epoch'+str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)

            # Save log
            torch.save(trlog, os.path.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))

        writer.close()

    def get_tacc(self, logits):
        one_hot_label = torch.cat((one_hot_encode(5, torch.arange(5)), torch.zeros(75,5)), dim=0).cuda()
        logits = torch.sigmoid(logits)*(1-torch.eye(80).cuda())
        label = torch.mm(logits, one_hot_label)
        pre_label = torch.mm(logits, label).max(dim=-1)[-1]
        acc = torch.eq(pre_label[5:], torch.arange(5).repeat(15).cuda()).float().mean().item()
        return acc

    def eval(self):
        # Load the logs
        trlog = torch.load(os.path.join(self.args.save_path, 'trlog'))

        # Load meta-test set
        test_set = Dataset(self.args.dataset_dir, 'test')
        sampler = Categories_Sampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

        # Set test accuracy recorder
        test_acc_record = np.zeros((600,))
        #print(self.model.gnn.state_dict()['last_edge_update.0.weight'])
        
        # Load model for meta-test phase
        if self.args.eval_weights is not None:
            self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.args.save_path, 'max_acc' + '.pth'))['params'])

        # Set model to eval mode
        self.model.eval()

        # Set accuracy averager
        ave_acc = Averager()

        # Generate labels
        label = torch.arange(self.args.way).repeat(self.args.val_query+self.args.shot).type(torch.cuda.LongTensor)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot).type(torch.cuda.LongTensor)
        label_edge = 1-label2edge(label)    
        # Start meta-test
        for i, batch in enumerate(loader, 1):
            data, _ = [_.cuda() for _ in batch]
            
            k = self.args.way * self.args.shot
            data_shot, data_query = data[:k], data[k:]
            logits, edge = self.model((data_shot, label_shot, data_query, label_edge))
            # t = torch.cat((edge.view(-1,1).detach(), label_edge.view(-1,1).detach()),dim=-1).cpu().numpy()
            # np.save('5s_tt_edge.npy',t)

            acc = count_acc(logits, label[25:])
            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            if i % 100 == 0:
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
            
        # Calculate the confidence interval, update the logs
        m, pm = compute_confidence_interval(test_acc_record)
        print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))

    def load_fea_extractor(self):
        pretrained_dict = torch.load(self.args.init_weights)['params']
        pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
        # print(pretrained_dict.keys())
        self.model_dict.update(pretrained_dict)
        self.model.load_state_dict(self.model_dict)   

    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), os.path.join(self.args.save_path, name + '.pth'))        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['miniImageNet', 'tieredImageNet']) # Dataset
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--num_gpus', type=int, default=1) # number of GPUs
    parser.add_argument('--dataset_dir', type=str, default='./data/mini/') # Dataset folder

    parser.add_argument('--max_epoch', type=int, default=150) # Epoch number for meta-train phase
    parser.add_argument('--batch_size', type=int, default=100) # The number for different tasks used for meta-train
    parser.add_argument('--shot', type=int, default=1) # Shot number, how many samples for one class in a task
    parser.add_argument('--way', type=int, default=5) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=15) # The number of training samples for each class in a task
    parser.add_argument('--val_query', type=int, default=15) # The number of test samples for each class in a task
    parser.add_argument('--lr1', type=float, default=0.0001) # Learning rate for SS weights
    parser.add_argument('--lr2', type=float, default=0.001) # Learning rate for FC weights
    parser.add_argument('--base_lr', type=float, default=0.01) # Learning rate for the inner loop
    parser.add_argument('--update_step', type=int, default=150) # The number of updates for the inner loop
    parser.add_argument('--reduce_step', type=int, default=10) # The number of epochs to reduce the meta learning rates
    parser.add_argument('--gamma', type=float, default=0.5) # Gamma for the meta-train learning rate decay
    parser.add_argument('--init_weights', type=str, default='pre_train_6022.pth') # The pre-trained weights for meta-train phase
    parser.add_argument('--eval_weights', type=str, default=None) # The meta-trained weights for meta-eval phase
    parser.add_argument('--exp_name', type=str, default='exp1') # Additional label for meta-train
    args = parser.parse_args()
    pprint(vars(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = FGNN_trainer(args)
    trainer.train()
    trainer.eval()
