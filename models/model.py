import jittor
from jittor import nn, Module, models
import pygmtools as pygm
from utils import *
from configs import *
pygm.BACKEND = 'jittor'

class CNNNet(Module):
    def __init__(self, VGG_pretrained_model):
        super(CNNNet, self).__init__()

        vgg16_module = models.vgg16_bn(False)

        self.node_layers = nn.Sequential(*[_ for _ in list(vgg16_module.features)[:31]])
        self.edge_layers = nn.Sequential(*[_ for _ in list(vgg16_module.features)[31:38]])
        if (not VGG_pretrained_model == None):    
            self.load_state_dict(jittor.load(VGG_pretrained_model))
    def execute(self, inp_img):
        feat_local = self.node_layers(inp_img)
        feat_global = self.edge_layers(feat_local)
        return feat_local, feat_global

class GMNET(Module):
    def __init__(self, VGG_pretrained_model=None):
        super(GMNET, self).__init__()
        self.gm_net = pygm.utils.get_network(pygm.pca_gm, pretrain = False)
        self.cnn = CNNNet(VGG_pretrained_model)
    
    def execute(self, img1, img2, kpts1, kpts2, A1, A2):
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)
        feat1_local = l2norm(feat1_local)
        feat2_local = l2norm(feat2_local)
        feat1_global = l2norm(feat1_global)
        feat2_global = l2norm(feat2_global)

        # upsample feature map
        feat1_local_upsample = nn.interpolate(feat1_local, obj_resize)
        feat2_local_upsample = nn.interpolate(feat2_local, obj_resize)
        feat1_global_upsample = nn.interpolate(feat1_global, obj_resize)
        feat2_global_upsample = nn.interpolate(feat2_global, obj_resize)
        feat1_upsample = jt.concat((feat1_local_upsample, feat1_global_upsample), dim = 1)
        feat2_upsample = jt.concat((feat2_local_upsample, feat2_global_upsample), dim = 1)

        # assign node features
        rounded_kpts1 = jt.round(kpts1).long()
        rounded_kpts2 = jt.round(kpts2).long()

        node1 = jt.stack([feat1_upsample[i, :, rounded_kpts1[i][1], rounded_kpts1[i][0]].t() for i in range(feat1_upsample.shape[0])])
        node2 = jt.stack([feat2_upsample[i, :, rounded_kpts2[i][1], rounded_kpts2[i][0]].t() for i in range(feat2_upsample.shape[0])])
      
        X = pygm.pca_gm(node1, node2, A1, A2, network= self.gm_net)
        return X
