from fastcore.all import *
from fastai.vision.all import *
from fastai.vision import *

class CELossFlat:
    "Cross Entropy Loss with flattening"
    def __init__(self, axis=1):
        store_attr()
        self.loss_func = CrossEntropyLossFlat(axis=axis)
        
    def __call__(self, pred, targ):
        return self.loss_func(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)

class CEDiceLoss:
    "Dice and CE combined"
    def __init__(self, axis=1, smooth=1., alpha=1.):
        store_attr()
        self.ce = CrossEntropyLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)
        
    def __call__(self, pred, targ):
        return self.ce(pred, targ) + self.alpha * self.dice_loss(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)

class FocalDiceLoss:
    "Dice and Focal combined"
    def __init__(self, axis=1, smooth=1., alpha=1.):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss =  DiceLoss(axis, smooth)
        
    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)