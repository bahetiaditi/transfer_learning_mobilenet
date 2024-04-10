# models/mobilenet.py
from torchvision.models import mobilenet_v2 as mobilenet_v2_pretrained

def mobilenet_v2(pretrained=True):
    if pretrained:
        model = mobilenet_v2_pretrained(pretrained=True).features
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = mobilenet_v2_pretrained(pretrained=False).features
    return model