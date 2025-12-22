import torch


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
    
    
    # iterate through the dataloader
    for batch in dataloader:
        X, Y, puzzle_ids = batch
        # apply pin memory and device, non_blocking
        if device.type != 'cpu':
            X = X.pin_memory().to(device, non_blocking=True)
            Y = Y.pin_memory().to(device, non_blocking=True)
            puzzle_ids = puzzle_ids.pin_memory().to(device, non_blocking=True)
        else:
            X = X.to(device)
            Y = Y.to(device)
            puzzle_ids = puzzle_ids.to(device)
            
        with torch.inference_mode():
            mask = (Y != IGNORE_LABEL_ID)
            logits, loss = model(X, puzzle_ids, Y, test_mode=True)
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == Y) & mask
            correct_pixels += correct.sum().item()
            total_pixels += mask.sum().item()
            exact_accuracy = correct.sum(-1) == mask.sum(-1)
            correct_puzzles += exact_accuracy.sum().item()
            total_puzzles += Y.size(0)
            losses += loss.item() * Y.size(0)  # sum up batch loss
            
    out['loss'] = losses / total_puzzles
    out['pixel_accuracy'] = correct_pixels / total_pixels
    out['exact_accuracy'] = correct_puzzles / total_puzzles
    
    return out