from torch import nn


class WDGRLNet(nn.Module):
    def __init__(self, generator, classifier, critic):
        super(WDGRLNet, self).__init__()
        self.g = generator
        self.c = classifier
        self.w = critic

    def gen(self, x, src=True):
        return self.g(x)

    def critic(self, h):
        return self.w(h)

    def classif(self, h):
        return self.c(h)

    def w_params(self):
        return list(self.w.parameters())
    
    def c_params(self):
        return list(self.c.parameters())

    def gs_params(self):
        return list(self.g.parameters())

    def gt_params(self):
        return list(self.g.parameters())


class MTWdgrlNet(WDGRLNet):
    def __init__(self, src_generator, trg_generator, classifier, critic):
        super(MTWdgrlNet, self).__init__(None, classifier, critic)
        self.gs = src_generator
        self.gt = trg_generator

    def gen(self, x, src=True):
        h = self.gs(x) if src else self.gt(x)
        return h

    def gs_params(self):
        theta_gs = self.gs.parameters()
        return list(theta_gs)

    def gt_params(self):
        theta_gt = self.gt.parameters()
        return list(theta_gt)
