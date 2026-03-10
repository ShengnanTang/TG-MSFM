import torch
from Utils.io_utils import instantiate_from_config
import os


def _collate(batch):
    if isinstance(batch[0], tuple):
        # 自动适配任意元素数的tuple（建议！）
        cols = list(zip(*batch))
        return tuple(torch.stack(col) for col in cols)
    return torch.stack(batch)

use_cuda = torch.cuda.is_available()
num_workers = 4 if use_cuda else 0
pin_memory  = True if use_cuda else False

def build_dataloader(config, args=None):
    batch_size = config['dataloader']['batch_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['train_dataset']['params']['output_dir'] = args.save_dir
    dataset = instantiate_from_config(config['dataloader']['train_dataset'])

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             sampler=None,
                                             drop_last=jud,
                                             collate_fn=_collate)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': dataset
    }

    return dataload_info

def build_dataloader_cond(config, args=None):
    batch_size = config['dataloader']['sample_size']
    config['dataloader']['test_dataset']['params']['output_dir'] = args.save_dir
    if args.mode == 'infill':
        config['dataloader']['test_dataset']['params']['missing_ratio'] = args.missing_ratio
    elif args.mode == 'predict':
        config['dataloader']['test_dataset']['params']['predict_length'] = args.pred_len
    test_dataset = instantiate_from_config(config['dataloader']['test_dataset'])

    dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             sampler=None,
                                             drop_last=False,
                                             collate_fn=_collate)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': test_dataset
    }

    return dataload_info


if __name__ == '__main__':
    pass

