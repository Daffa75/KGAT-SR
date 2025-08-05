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
        self.n_node = self.n_items
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=args.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

        self.model_init()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def model_init(self):
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

        self.final_activation = nn.ReLU()
        self.activation_sigmoid = nn.Sigmoid()

    def get_entitygat(self, h_items, h_attrs, t_item):
        # Get embeddings for historical items, attributes, and the target item
        h_item_embs = self.item_emb_table(h_items)
        h_attr_embs = self.item_emb_table(h_attrs)
        t_item_emb = self.item_emb_table(t_item)

        # Reshape attribute embeddings and sum them up
        h_attr_embs = h_attr_embs.view(h_attrs.shape[0], h_attrs.shape[1], h_attrs.shape[2], -1)
        h_attr_embs = torch.sum(h_attr_embs, dim=2)

        # Apply dropout to item and attribute embeddings
        h_item_embs = self.emb_dropout_layer(h_item_embs)
        h_attr_embs = self.emb_dropout_layer(h_attr_embs)

        # Calculate attention scores
        # Project item and attribute embeddings to a common space
        q_item = self.linear_attention_1(h_item_embs)
        q_attr = self.linear_attention_1(h_attr_embs)
        
        # Add the projected embeddings
        q = q_item + q_attr
        
        # Calculate attention weights
        alpha = self.linear_attention(q)
        alpha = F.softmax(alpha, dim=1)

        # Apply attention weights to attribute embeddings
        if self.aggregate == 'concat':
            # Concatenate item and attribute embeddings
            z = torch.cat([h_item_embs, h_attr_embs], dim=-1)
        else:
            # Add item and attribute embeddings
            z = h_item_embs + h_attr_embs
            
        # Transform the aggregated embedding
        z = self.linear_attention_transform(z)
        
        # Compute the final entity-aware item embedding
        entity_aware_item_emb = torch.sum(alpha * z, dim=1)

        return entity_aware_item_emb
    
    def forward(self, items, A, h_items, h_attrs, t_item):
        # Get session graph embeddings
        hidden = self.embedding(items)
        hidden = self.gnn(A, hidden)

        # Get knowledge-enhanced entity embeddings
        entitygat = self.get_entitygat(h_items, h_attrs, t_item)

        return hidden, entitygat

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

def forward(model, i, data):
    # The get_slice method now returns all the data we need
    alias_inputs, A, items, mask, targets, h_items, h_attrs, t_item = data.get_slice(i)

    # Move data to GPU
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    h_items = trans_to_cuda(torch.from_numpy(h_items).long())
    h_attrs = trans_to_cuda(torch.from_numpy(h_attrs).long())
    t_item = trans_to_cuda(torch.from_numpy(t_item).long())

    # Get hidden states from the model
    hidden = model(items, A)
    
    # Get the hidden state of the last item in each session
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    
    # Compute scores and return targets
    _, scores = model.compute_scores(seq_hidden, mask)
    return targets, scores

# Replace the existing train_test function in SRGATM.py
def train_test(model, train_data, test_data, batch, args):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        # The forward function now handles getting data and running the model
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
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
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask[i]):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr




