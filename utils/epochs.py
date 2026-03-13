import torch

from utils.const import IGNORE_LABEL_ID

def eval_epoch(config: dict,
               model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               device: torch.device) -> dict:
    out = {}
    model.eval()

    total_puzzles, total_pixels = 0, 0
    correct_puzzles, correct_pixels = 0, 0
    losses = 0.0

    # subsample the dataloader for faster evaluation
    sample_dataloader = torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
    )

    test_iters = config['logging'].get('eval_iters', 1)
    # fix B1: initialize iterator before loop (was NameError on first call)
    sample_iter = iter(sample_dataloader)

    if device.type != 'cpu':
        device_fn = lambda t: t.pin_memory().to(device, non_blocking=True)
    else:
        device_fn = lambda t: t.to(device)
    detach_fn = lambda t: t.detach().cpu()

    for _ in range(test_iters):
        try:
            batch = next(sample_iter)
        except StopIteration:
            sample_iter = iter(sample_dataloader)
            batch = next(sample_iter)

        X, Y, puzzle_ids = batch

        X = device_fn(X)
        Y = device_fn(Y)
        puzzle_ids = device_fn(puzzle_ids)

        with torch.inference_mode():
            logits, loss = model(X, puzzle_ids, Y, test_mode=True)

            # detach and move to cpu for metric calculations
            logits = detach_fn(logits)
            loss = detach_fn(loss)
            Y_cpu = detach_fn(Y)

        # fix B9: compute mask after detach to avoid device mismatch
        mask = (Y_cpu != IGNORE_LABEL_ID)
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == Y_cpu) & mask
        correct_pixels += correct.sum().item()
        total_pixels += mask.sum().item()
        exact_accuracy = correct.sum(-1) == mask.sum(-1)
        correct_puzzles += exact_accuracy.sum().item()
        total_puzzles += Y_cpu.size(0)
        losses += loss.item() * Y_cpu.size(0)

    out['loss'] = losses / total_puzzles if total_puzzles > 0 else 0.0
    out['accuracy'] = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    out['sequence_accuracy'] = correct_puzzles / total_puzzles if total_puzzles > 0 else 0.0

    return out
