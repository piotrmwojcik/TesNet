import os
import torch

class HeapPatch:
    def __init__(self, patch, filename, distance, mask_patch):
        self.patch = patch
        self.filename = filename
        self.mask_patch = mask_patch
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

