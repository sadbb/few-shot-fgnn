import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet
#torch.set_printoptions(threshold=100)

class BaseLearner(nn.Module):
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([5, self.z_dim*2]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(5))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None, edge=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        n, d = input_x.size()
        mask = (1-torch.eye(n)).cuda()
        dis = F.normalize(edge*mask, p=1, dim=-1)
        x = torch.mm(dis, input_x)
        # x = self.Acompute(input_x)
        net = F.linear(torch.cat((x, input_x), dim=-1), fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars

    def Acompute(self, node_feat):
        n, d = node_feat.size()
        mask = (1-torch.eye(n)).cuda()
        node_feat0 = node_feat.unsqueeze(1).repeat(1, n, 1)
        node_feat1 = node_feat0.transpose(0,1).contiguous()
        dis1 = F.pairwise_distance(node_feat0.view(-1, d), node_feat1.view(-1, d), p=2).view(n, n)
        dis =2* torch.sigmoid(dis1)-1
        dis = F.normalize(dis1*mask, p=1, dim=-1)
        return torch.mm(dis, node_feat)

class FGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        self.in_feature = 640
        self.GNN_classifier = BaseLearner(args, self.in_feature)
        self.encoder = ResNet()  
        self.relation_encoder = GraphNetwork(640)
        #self.fc = nn.Linear(320, 5)
    
    # def edge_forward(self,inp):
    #     data_shot, label_shot, data_query = inp
    
    #     # embedding_query = self.encoder(data_query)
    #     # embedding_shot = self.encoder(data_shot) 
    #     embedding_data = self.encoder(torch.cat((data_shot, data_query), dim=0)) # [100, 640]
    #     n, d = embedding_data.size()
    #     embedding_data = embedding_data.unsqueeze(1).repeat(1,n,1)
    #     embedding_data_ = embedding_data.transpose(0,1)
    #     x = torch.cat((embedding_data, embedding_data_), dim=-1)
    #     return self.classifier(x.view(-1, 2*d))

    def forward(self, inp):

        data_shot, label_shot, data_query, label_edge = inp
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
    
        embedding_data = torch.cat((embedding_shot,embedding_query), dim=0)
        embedding, edge = self.relation_encoder(embedding_data)
        top_mask = edge>0.7
        bottom_mask = edge<0.3
        edge_ = edge*(1-(top_mask+bottom_mask).float())+top_mask.float()
        sn = 5*self.args.shot
        logits = self.GNN_classifier(embedding_data,edge=edge_)[:sn]
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.GNN_classifier.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.GNN_classifier.parameters())))
        logits_q = self.GNN_classifier(embedding_data,edge=edge_)[sn:]

        for _ in range(1, self.update_step):
            logits = self.GNN_classifier(embedding_data, fast_weights, edge=edge_)[:sn]
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.GNN_classifier(embedding_data,fast_weights,edge=edge_)[sn:]
        
        return logits_q,edge
    
class GraphNetwork(nn.Module):
    def __init__(self, input_size=640, output_size=[]):
        super(GraphNetwork, self).__init__()
        output_size.insert(0, input_size)
        self.num_layers = len(output_size)-1
        for i in range(self.num_layers):
            edge_update = nn.Sequential(
                nn.Linear(output_size[i], output_size[i]*2),
                nn.BatchNorm1d(output_size[i]*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(output_size[i]*2, 1),
                nn.Sigmoid()
            )
            node_update = nn.Sequential(
                nn.Linear(output_size[i]*2, output_size[i+1]),
                nn.BatchNorm1d(output_size[i+1]),
                nn.LeakyReLU(inplace=True)
            )
            self.add_module('NodeUpdate{}'.format(i), node_update)
            self.add_module('EdgeUpdate{}'.format(i), edge_update)
        self.last_edge_update = nn.Sequential(
                nn.Linear(output_size[-1], output_size[-1]*2),
                nn.BatchNorm1d(output_size[-1]*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(output_size[-1]*2, 1),
                nn.Sigmoid()
            )
        self.mask = torch.eye(100).cuda()

    def forward(self, node_feat, adj_matrix=None):
        for i in range(self.num_layers):
            n, d = node_feat.size()
            node_feat0 = node_feat.unsqueeze(1).repeat(1, n, 1)
            node_feat1 = node_feat0.transpose(0,1).contiguous()
            dis = (self._modules['EdgeUpdate{}'.format(i)](torch.abs(node_feat0.view(-1, d)-node_feat1.view(-1,d))).squeeze().view(n,n)) * (1-self.mask)
            node_feat = torch.cat((node_feat, torch.mm(dis, node_feat/dis.sum(-1).unsqueeze(-1))), dim=-1)
            node_feat = self._modules['NodeUpdate{}'.format(i)](node_feat)
        n, d = node_feat.size()
        node_feat0 = node_feat.unsqueeze(1).repeat(1, n, 1)
        node_feat1 = node_feat0.transpose(0,1).contiguous()
        dis = self.last_edge_update(torch.abs(node_feat0.view(-1,d)-node_feat1.view(-1,d))).squeeze().view(n,n)
        return node_feat, dis
