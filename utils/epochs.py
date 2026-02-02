import torch
import tqdm

from utils.const import IGNORE_LABEL_ID

def eval_epoch(config: dict,
               model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               device: torch.device) -> dict:
    """

    """
    out = {}
    model.eval()
    
    total_puzzles, total_pixels = 0, 0
    correct_puzzles, correct_pixels = 0, 0
    losses = 0.0 # average of over all batches
    
    # we subsample the dataloader for faster evaluation
    sample_dataloader = torch.utils.data.DataLoader(dataloader.dataset,
                                       batch_size=dataloader.batch_size,
                                       shuffle=True,
                                       num_workers=dataloader.num_workers,
                                       pin_memory=dataloader.pin_memory)
    
    # iterate through the dataloader
    test_iters = config['logging'].get('eval_iters', 1)
    for _ in range(test_iters):
        try:
            batch = next(pbar_iter)
        except:
            pbar_iter = iter(sample_dataloader)
            batch = next(pbar_iter)
        X, Y, puzzle_ids = batch
        mask = (Y != IGNORE_LABEL_ID)
        # apply pin memory and device, non_blocking
        
        device_fn = None
        detach_fn = None
        if device.type != 'cpu':
            device_fn = lambda t: t.pin_memory().to(device, non_blocking=True)
            detach_fn = lambda t: t.detach().cpu()
        else:
            device_fn = lambda t: t.to(device)
            detach_fn = lambda t: t.detach().cpu()
        
        X = device_fn(X)
        Y = device_fn(Y)
        puzzle_ids = device_fn(puzzle_ids)
        
        with torch.inference_mode():
            logits, loss = model(X, puzzle_ids, Y, test_mode=True)
            
            # detach and move to cpu for metric calculations
            logits = detach_fn(logits)
            loss = detach_fn(loss)
            Y = detach_fn(Y)
            
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == Y) & mask
            correct_pixels += correct.sum().item()
            total_pixels += mask.sum().item()
            exact_accuracy = correct.sum(-1) == mask.sum(-1)
            correct_puzzles += exact_accuracy.sum().item()
            total_puzzles += Y.size(0)
            losses += loss.item() * Y.size(0)  # sum up batch loss
            
    out['loss'] = losses / total_puzzles
    out['accuracy'] = correct_pixels / total_pixels
    out['sequence_accuracy'] = correct_puzzles / total_puzzles
    
    return out