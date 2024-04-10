# utils/metrics.py
def calc_iou(preds, labels):
    smooth = 1e-6
    preds_bool = torch.sigmoid(preds) > 0.5
    labels_bool = labels > 0.5


    intersection = (preds_bool & labels_bool).float().sum((1, 2))
    union = (preds_bool | labels_bool).float().sum((1, 2))

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def calc_dice(preds, labels):
    smooth = 1e-6
    preds = torch.sigmoid(preds) > 0.5
    labels_bool = labels > 0.5
    intersection = (preds & labels_bool).float().sum((1, 2))
    dice = (2. * intersection + smooth) / (preds.sum((1, 2)) + labels_bool.sum((1, 2)) + smooth)

    return dice.mean()

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):

        outputs = torch.sigmoid(outputs)


        outputs_flat = outputs.view(-1)
        targets_flat = targets.view(-1)


        intersection = (outputs_flat * targets_flat).sum()
        total = (outputs_flat + targets_flat).sum()
        union = total - intersection

        IoU = (intersection + self.smooth) / (union + self.smooth)

        return 1 - IoU