def accuracy_segmentation(inp, target):
    target = target.squeeze(1)
    return (inp.argmax(dim=1) == target).float().mean()
