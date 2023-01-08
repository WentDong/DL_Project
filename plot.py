'''
This program is used to plot the matching result of a model together with the visualization of CNN feature.
You can use "path", "Eval_model_name", "plot_saving_path" to change the model and saving dir.
'''
import jittor as jt
import pygmtools as pygm
from configs import *
from models.model import GMNET
from dataset.dataloader import WillowObject
import os
from tqdm import *
from utils import *
from matplotlib.patches import ConnectionPatch # for plotting matching result
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as PCAdimReduc
from args import *
pygm.BACKEND = 'jittor'

def get_feat(model, img1, img2):
    with jt.no_grad():
        feat1_local, feat1_global = model.cnn(img1)
        feat2_local, feat2_global = model.cnn(img2)
        feat1_local = l2norm(feat1_local)
        feat2_local = l2norm(feat2_local)
        feat1_global = l2norm(feat1_global)
        feat2_global = l2norm(feat2_global)
        feat1_local_upsample = nn.interpolate(feat1_local, obj_resize)
        feat2_local_upsample = nn.interpolate(feat2_local, obj_resize)
        feat1_global_upsample = nn.interpolate(feat1_global, obj_resize)
        feat2_global_upsample = nn.interpolate(feat2_global, obj_resize)
        feat1_upsample = jt.concat((feat1_local_upsample, feat1_global_upsample), dim = 1)
        feat2_upsample = jt.concat((feat2_local_upsample, feat2_global_upsample), dim = 1)
        num_features = feat1_upsample.shape[1]
        pca_dim_reduc = PCAdimReduc(n_components=3, whiten=True)
        feat_dim_reduc = pca_dim_reduc.fit_transform(
            np.concatenate((
                feat1_upsample.permute(0, 2, 3, 1).reshape(-1, num_features).numpy(),
                feat2_upsample.permute(0, 2, 3, 1).reshape(-1, num_features).numpy()
            ), axis=0)
        )
        feat_dim_reduc = feat_dim_reduc / np.max(np.abs(feat_dim_reduc), axis=0, keepdims=True) / 2 + 0.5
        feat1_dim_reduc = feat_dim_reduc[:obj_resize[0] * obj_resize[1], :]
        feat2_dim_reduc = feat_dim_reduc[obj_resize[0] * obj_resize[1]:, :]
        return feat1_dim_reduc, feat2_dim_reduc

if __name__ == '__main__':
    args = get_args("PCA_GM")
    jt.flags.use_cuda = jt.has_cuda
    model = GMNET()
    PlotLoader = WillowObject(Train = False, Shuffle = args.Shuffle, Eval = False).set_attrs(batch_size = 1, shuffle = False)
    
    model.load_state_dict(jt.load(os.path.join(args.path, args.Eval_model_name)))

    loop = tqdm(enumerate(PlotLoader), total = len(PlotLoader))

    if not os.path.exists(args.plot_saving_path):
        os.mkdir(args.plot_saving_path)

    for index, (ori_img1, ori_img2, img1, img2, kpts1, kpts2, A1, A2) in loop:
        X = model(img1, img2, kpts1, kpts2, A1, A2)
        feat1, feat2 = get_feat(model, img1, img2)

        X = jt.squeeze(X,0)
        X = pygm.hungarian(X)
        plt.figure(figsize=(8, 4))
        plt.suptitle('Image Matching Result by PCA-GM')
        img1 = jt.squeeze(img1, 0)
        img2 = jt.squeeze(img2, 0)
        kpts1 = jt.squeeze(kpts1, 0)
        kpts2 = jt.squeeze(kpts2, 0)
        ori_img1 = jt.squeeze(ori_img1, 0).numpy()
        ori_img2 = jt.squeeze(ori_img2, 0).numpy()

        A1 = jt.squeeze(A1, 0)
        A2 = jt.squeeze(A2, 0)
        ax1 = plt.subplot(1, 2, 1)

        plot_image_with_graph(ori_img1, kpts1, A1)
        plt.imshow(feat1.reshape(obj_resize[1], obj_resize[0], 3), alpha=0.5)

        ax2 = plt.subplot(1, 2, 2)
        plot_image_with_graph(ori_img2, kpts2, A2)
        plt.imshow(feat2.reshape(obj_resize[1], obj_resize[0], 3), alpha=0.5)
        idx, _ = jt.argmax(X, dim=1)
        for i in range(X.shape[0]):
            j = idx[i].item()
            con = ConnectionPatch(xyA=kpts1[:, i], xyB=kpts2[:, j], coordsA="data", coordsB="data",
                                axesA=ax1, axesB=ax2, color="red" if i != j else "green")
            plt.gca().add_artist(con)
        plt.savefig(os.path.join(args.plot_saving_path, f"Test_{index}_matching.png"))
        