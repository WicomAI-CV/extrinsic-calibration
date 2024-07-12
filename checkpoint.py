import torch

def save_checkpoint(state, filename="ViT_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    last_epoch = checkpoint["epoch"]
    last_epoch_loss = checkpoint["loss"]
    last_best_error_t = checkpoint["trans_error"] 
    last_best_error_r = checkpoint["rot_error"]
    
    return model, last_epoch, last_epoch_loss, last_best_error_t, last_best_error_r