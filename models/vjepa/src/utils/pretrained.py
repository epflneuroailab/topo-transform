import logging

import torch

def load_checkpoint(r_path, module, is_head=False, remove_module=False):
    try:
        if is_head:
            checkpoint = torch.load(r_path, map_location=torch.device('cpu'))["classifier"]
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            msg = module.load_state_dict(checkpoint, strict=True)
            print(f'Loaded pretrained component with msg: {msg}')
        else:
            checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
            if remove_module:
                checkpoint = {k.replace("module.backbone.", ""): v for k, v in checkpoint["target_encoder"].items()}
            msg = module.load_state_dict(checkpoint, strict=False)
            print(f'Loaded pretrained component with msg: {msg}')
    except Exception as e:
        print(f'Encountered exception when loading checkpoint {e}')
        raise Exception()
    
    return module