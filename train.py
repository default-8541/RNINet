from model.RNINet import *
from train_lab.train_base import *
from data.create_dataset import *
from loss import *
import random
from easydict import EasyDict
import json
import argparse
import os

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option JSON file.')
    args = parser.parse_args()
    _,config_dict=get_config_from_json(args.opt)
    #print(config_dict)
    args,dataset_opt=config_dict["args"],config_dict["dataset_opt"]
    
    basic_dir="/home/debug/"
    if os.path.exists(basic_dir)==False:
        os.mkdir(basic_dir)
        
    args["model_path"]=basic_dir+"save_model/"
    args["train_logger"]=basic_dir+"train"
    args["test_logger"]=basic_dir+"test"
    #args["device"]="cuda:1"
    args["stop"]=1000000
    
    with open(basic_dir+"config.json", 'w') as f:
        json.dump(config_dict, f)
    
    print(config_dict)

    net=RNINet(apply_prob=0.5,ep=1e-6,in_nc=3, out_nc=3,act_mode='BR')
    device=args["device"]
    net.to(device)
    
    seed=args["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.seed_all()
    
    train_loader, test_loader=create_dataset(dataset_opt)
    
    logger_info(args["train_logger"],args["train_logger"]+".log")
    logger_info(args["test_logger"],args["test_logger"]+".log")
    
    if os.path.exists(args["model_path"])==False:
        os.mkdir(args["model_path"])
    
    loss_fn=CharbonnierLoss()
    
    train_backbone(net, train_loader, test_loader, args, loss_fn)

    
if __name__ == '__main__':
    main()
