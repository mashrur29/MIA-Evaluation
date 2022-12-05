import os

def get_mode_dir(dataset_name, isDefense):
    root = 'trained_models'
    if isDefense:
        dataset = dataset_name + '_with_defense/model_best.pth.tar'
        return os.path.join(root, dataset)
    else:
        dataset = dataset_name + '_no_defense/model_best.pth.tar'
        return os.path.join(root, dataset)