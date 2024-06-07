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
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size of cnn, in early stage.')
    parser.add_argument('--final_hidden_size', type=int, default=64,
                        help='Hidden size of cnn, in early stage.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')

    # Training parameters
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
