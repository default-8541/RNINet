import sys

sys.path.append("..")
#from args_file import dataset_opt
from torch.utils.data import DataLoader


def define_Dataset(dataset_opt):
    if dataset_opt["dataset_type"]=="dncnn":
        from data.dataset_dncnn import DatasetDnCNN as D
    elif dataset_opt["dataset_type"]=="jpeg":
        from data.dataset_jpeg import DatasetJPEG as D
    elif dataset_opt["dataset_type"]=="sr":
        from data.dataset_sr import DatasetSR as D
    elif dataset_opt["dataset_type"]=="blindsr":
        from data.dataset_blindsr import DatasetBlindSR as D
    elif dataset_opt["dataset_type"]=="dncnn_cnn":
        from data.dataset_dncnn_cnn import DatasetDnCNN as D
    elif dataset_opt["dataset_type"]=="jpeg_jpeg":
        from data.dataset_jpeg_jpeg import DatasetJPEG as D
    elif dataset_opt["dataset_type"]=="poisson":
        from data.dataset_dncnn_poisson import DatasetDnCNN as D
    return D(dataset_opt)


def create_dataset(dataset_opt):
    for phase, d_opt in dataset_opt.items():
        if phase == "train":
            train_set = define_Dataset(d_opt)
            train_loader = DataLoader(train_set,
                                      batch_size=d_opt['dataloader_batch_size'],
                                      shuffle=d_opt['dataloader_shuffle'],
                                      num_workers=d_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == "test":
            test_set = define_Dataset(d_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)

    return train_loader, test_loader
