import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import logging
from torch.nn.utils.rnn import pad_sequence


from Data_Loader.Config import parse_args
from Data_Loader.Data_Module import DataModule, collate_fn
from Data_Loader.data_preprocessing import Data_Preprocess

from Models.CNN import TimeSeriesCNN
from Models.Decision_Tree import TimeSeriesDT
from Trainer.trainCNN import train_timeseries_Conv


logger = logging.getLogger(__name__)


### Run the model from here
if __name__ == '__main__':

    # Get all the parameters from parameters
    args = parse_args()
    longest_raw_data = 0
    # First, load the data from the dataset
    OK_raw_data, l1= DataModule(args.datasetOK_root,args)._load_data()
    NOK_raw_data, l2= DataModule(args.datasetNOK_root, args)._load_data()
    longest_raw_data = l1 if l1 >= l2 else l2
    print(l1,l2)
    #print(OK_raw_data)
    #print(len(OK_raw_data))
    #Second, preprocessing the data
    OK_dataset = Data_Preprocess(OK_raw_data).process_signal(OK_raw_data,args.datasetOK_root)
    #print(OK_dataset.tensors[0].shape)
    NOK_dataset = Data_Preprocess(NOK_raw_data).process_signal(NOK_raw_data,args.datasetNOK_root)
    # # Concatenate the features and labels
    combined_features = torch.cat([OK_dataset.tensors[0], NOK_dataset.tensors[0]], dim=0)
    combined_labels = torch.cat([OK_dataset.tensors[1], NOK_dataset.tensors[1]], dim=0)
    # # Create a new combined TensorDataset
    combined_dataset = TensorDataset(combined_features, combined_labels)
    combined_dataloader = DataLoader(combined_dataset, batch_size=args.train_batch_size, collate_fn = collate_fn, shuffle=True)



    # Third, train the CNN to get the model
    #train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    CNNmodel = train_timeseries_Conv(combined_dataloader,device,args,longest_raw_data)
    embed_features = CNNmodel.get_embed(combined_features)

    # Put the embedding feature into decision tree to classify
    result = TimeSeriesDT([embed_features,combined_labels],args)


