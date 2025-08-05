import model
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import math
import numpy as np
import datetime

device = torch.device('cuda:0')
class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class SRGAT(nn.Module):
    def __init__(self, args, n_items, n_rels, D_node, adj_entity, adj_relation):
        super().__init__()
        self.n_items = n_items
        self.n_rels = n_rels

        self.emb_size = args.emb_size
        self.hidden_size = args.emb_size
        self.n_layers = args.n_layers
        self.emb_dropout = args.emb_dropout
        self.hidden_dropout = args.hidden_dropout
        self.gradient_clip = args.gradient_clip

        self.order = args.order
        self.neibor_size = args.neibor_size
        self.attention = args.attention
        self.aggregate = args.aggregate

        self.D_node = D_node
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.batch_size = args.batchSize
        self.nonhybrid = args.nonhybrid
        self.n_node = self.n_items # Correctly set n_node from n_items
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=args.step)
        
        # --- All model layers are now correctly initialized within SRGAT ---
        self.rel_emb_table = nn.Embedding(self.n_rels, self.emb_size, padding_idx=0)
        self.item_emb_table = nn.Embedding(self.n_items, self.emb_size, padding_idx=0)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # local interest
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # global perference
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_attention_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_attention = nn.Linear(self.hidden_size, 1, bias=True)
        self.linear_attr = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        if self.aggregate == 'concat':
            self.linear_attention_transform = nn.Linear(self.hidden_size * 2, self.hidden_size)
        else:
            self.linear_attention_transform = nn.Linear(self.hidden_size, self.hidden_size)

        self.gru = nn.GRU(self.emb_size, self.hidden_size, self.n_layers,
                          dropout=self.hidden_dropout, batch_first=True)
        self.emb_dropout_layer = nn.Dropout(self.emb_dropout)
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_entitygat(self, h_items, h_attrs):
        h_item_embs = self.item_emb_table(h_items)
        h_attr_embs = self.item_emb_table(h_attrs)
        h_attr_embs = h_attr_embs.view(h_attrs.shape[0], h_attrs.shape[1], h_attrs.shape[2], -1)
        h_attr_embs = torch.sum(h_attr_embs, dim=2)
        h_item_embs = self.emb_dropout_layer(h_item_embs)
        h_attr_embs = self.emb_dropout_layer(h_attr_embs)
        q_item = self.linear_attention_1(h_item_embs)
        q_attr = self.linear_attention_1(h_attr_embs)
        q = q_item + q_attr
        alpha = self.linear_attention(q)
        alpha = F.softmax(alpha, dim=1)
        if self.aggregate == 'concat':
            z = torch.cat([h_item_embs, h_attr_embs], dim=-1)
        else:
            z = h_item_embs + h_attr_embs
        z = self.linear_attention_transform(z)
        entity_aware_item_emb = torch.sum(alpha * z, dim=1)
        return entity_aware_item_emb

    # --- `compute_scores` is now a method of SRGAT ---
    def compute_scores(self, hidden, mask, entitygat):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)

        if not self.nonhybrid:
            session_embedding = self.linear_transform(torch.cat([a, ht], 1))
            final_embedding = session_embedding + entitygat
        else:
            final_embedding = a

        b = self.embedding.weight[1:]
        scores = torch.matmul(final_embedding, b.transpose(1, 0))
        return scores

    # --- The `forward` method now performs the complete model pass ---
    def forward(self, alias_inputs, A, items, mask, h_items, h_attrs):
        hidden = self.embedding(items)
        hidden = self.gnn(A, hidden)
        
        entitygat = self.get_entitygat(h_items, h_attrs)
        
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        
        return self.compute_scores(seq_hidden, mask, entitygat)

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def train_test(model, train_data, test_data, batch, args):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        
        # Get data and move to GPU
        alias_inputs, A, items, mask, targets, h_items, h_attrs, _ = train_data.get_slice(i)
        alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
        items = trans_to_cuda(torch.Tensor(items).long())
        A = trans_to_cuda(torch.Tensor(A).float())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        h_items = trans_to_cuda(torch.from_numpy(h_items).long())
        h_attrs = trans_to_cuda(torch.from_numpy(h_attrs).long())
        targets = trans_to_cuda(torch.Tensor(targets).long())

        # Call the model's forward pass directly
        scores = model(alias_inputs, A, items, mask, h_items, h_attrs)
        
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        alias_inputs, A, items, mask, targets, h_items, h_attrs, _ = test_data.get_slice(i)
        alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
        items = trans_to_cuda(torch.Tensor(items).long())
        A = trans_to_cuda(torch.Tensor(A).float())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        h_items = trans_to_cuda(torch.from_numpy(h_items).long())
        h_attrs = trans_to_cuda(torch.from_numpy(h_attrs).long())

        scores = model(alias_inputs, A, items, mask, h_items, h_attrs)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        
        for score, target in zip(sub_scores, targets):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr



