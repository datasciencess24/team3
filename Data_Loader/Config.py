import argparse



def parse_args():

    parser = argparse.ArgumentParser()

    ######### Parameters for the File path
    parser.add_argument('--datasetOK_root', type=str, default='./D/OK_Measurements',
                        help='Data path for all the OK measurements')
    parser.add_argument('--datasetNOK_root', type=str, default='./D/NOK_Measurements',
                        help='Data path for all the NOK measurements')
    parser.add_argument('--output_dir', type=str, default='./Output',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')

    ############# Parameters for the Basic Configeration
    parser.add_argument('--cuda_id', type=str, default='1',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=2022,
                        help='random seed for initialization')
    # CNN
    parser.add_argument('--gat', action='store_true',
                        help='GAT')
    parser.add_argument('--gat_our', action='store_true',
                        help='GAT_our')
    parser.add_argument('--gat_attention_type', type=str, choices=['linear', 'dotprod', 'gcn'], default='dotprod',
                        help='The attention used for gat')
    parser.add_argument('--embedding_dim', type=int, default=43,
                        help='Dimension of glove embeddings')
    parser.add_argument('--relation_num', type=int, default=3,
                        help='Total number of relation between nodes')
    parser.add_argument('--dep_relation_embed_dim', type=int, default=43,
                        help='Dimension for dependency relation embeddings.')
    parser.add_argument('--hidden_size', type=int, default=43,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--final_hidden_size', type=int, default=43,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')

    # Training parameters
    # parser.add_argument("--per_gpu_train_batch_size", default=60, type=int,
    #                     help="Batch size per GPU/CPU for training.")
    # parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
    #                     help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--train_batch_size", default=5, type=int,
                        help="Batch Size for trainning")
    parser.add_argument("--splits", default=0.6, type=int,
                        help="Train and Test split")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--timeseries_Conv_model', type=str, default='',
                        help="if it has timeseries_Conv_model.")

    return parser.parse_args()



# class Config():
#     def __init__(self):
#         # data_path = '../data/data1/train/power.csv'
#         self.num_embeddings = 23 # number of catergory of the embedidng
#         self.embedding_dim = 32
#         #self.n_samples = 58
#         #self.timestep = 14  # 时间步长，就是利用多少时间窗口
#         self.batch_size = 5  # 批次大小 since for trainning we only have 58 data
#         self.feature_size = 46  # 每个步长对应的特征数量，这里只使用1维，每天的风速
#         self.hidden_size = 56  # 隐层大小
#         self.output_size = 3  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
#         self.num_layers = 1  # cnn的层数
#         self.num_epochs = 10  # 迭代轮数
#         self.best_loss = 0  # 记录损失
#         self.learning_rate = 0.00003  # 学习率
#         self.dropout = 0.01
#         self.model_name = 'cnn'
#         self.save_path = './{}.pth'.format(model_name)  # 最优模型保存路径