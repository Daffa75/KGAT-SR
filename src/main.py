import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from torch.utils.data import Dataset, DataLoader

import utils
from SRGATM import *
from kg import KGraph  # Import the KGraph class
device = torch.device('cuda:0')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='movielens', help='movielens')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--emb_size', type=int, default=100)
parser.add_argument('--emb_dropout', type=int, default=0.1)
parser.add_argument('--hidden_dropout', type=int, default=0)


parser.add_argument('--hist_min_len', type=int, default=10)
parser.add_argument('--hist_max_len', type=int, default=20)
parser.add_argument('--neibor_size', type=int, default=4)

parser.add_argument('--attr_size', type=int, default=2)
parser.add_argument('--attention', type=bool, default=True)
parser.add_argument('--aggregate', type=str, default='concat')
parser.add_argument('--n_workers', type=int, default=4)
parser.add_argument('--gradient_clip', type=int, default=0)
parser.add_argument('--n_neg', type=int, default=500)
parser.add_argument('--order', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--valid', type=bool, default=False)
opt = parser.parse_args()
# opt.dataset = 'movielens'
print(opt)


def main():
    train_data_sessions = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data_sessions, valid_data_sessions = split_validation(train_data_sessions, opt.valid_portion)
        test_data_sessions = valid_data_sessions
    else:
        test_data_sessions = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    # 1. Create the Knowledge Graph object
    kg = KGraph(dataset=opt.dataset, attr_size=opt.attr_size)
    n_node = kg.n_entity # Get number of nodes from the KG

    # 2. Pass the knowledge graph object (kg) to the Data class constructor
    train_data = Data(train_data_sessions, kg=kg, shuffle=True)
    test_data = Data(test_data_sessions, kg=kg, shuffle=False)

    # The SRGAT model requires several KG-related arguments for initialization.
    # The `Loader` class is a convenient way to get them, even if we don't use its data loaders directly.
    loader = utils.Loader(opt)
    
    model = trans_to_cuda(SRGAT(opt, n_node, loader.n_relation, 
                                torch.Tensor(loader.D_node).to(device), 
                                torch.Tensor(loader.adj_entity).long().to(device), 
                                torch.Tensor(loader.adj_relation).long().to(device)))



    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data, opt.batchSize, opt)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1

        # --- Add this block to save the model ---
        if flag == 1:
            torch.save(model.state_dict(), f'{opt.dataset}_best_model.pth')
            print('Saving best model state to disk...')
        # ----------------------------------------
        
        print('Best Result:')
        print('\tHR@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
        best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
