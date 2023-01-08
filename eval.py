'''
This is the evaluation program
Feel free to use the function "evaluation".
'''
import jittor as jt
import pygmtools as pygm
from models.model import GMNET
from dataset.dataloader import WillowObject
import os
from tqdm import *
from args import *
pygm.BACKEND = 'jittor'

def evaluation(TestLoader, model, args, rm_cache = False):
    loop = tqdm(enumerate(TestLoader), total=len(TestLoader) // args.batch_size)
    pred = []
    for index, (img1, img2, kpts1, kpts2, A1, A2, idx1, idx2, cls1, cls2) in loop:
        # print("A")
        with jt.no_grad():
            X = model(img1, img2, kpts1, kpts2, A1, A2)

            for i in range(len(X)):
                Xi = pygm.hungarian(X[i])
                predict = {'ids': [idx1[i], idx2[i]], 'cls': cls1[i], 'perm_mat': Xi.numpy()}
                pred.append(predict)

    TestLoader.eval(pred, TestLoader.cls, verbose=True, rm_gt_cache=rm_cache)



if __name__ == '__main__':
    args = get_args("PCA_GM")
    jt.flags.use_cuda = jt.has_cuda 
    model = GMNET()
    TestLoader = WillowObject(Train=False, Shuffle=False, Eval=True).set_attrs(batch_size=args.batch_size,
                                                                               shuffle=False)
    model.load_state_dict(jt.load(os.path.join(args.path, args.Eval_model_name)))
    # model.load_state_dict(jt.load("./pygmtools-pretrain-models/pca_gm_willow_jittor.pt"))

    evaluation(TestLoader, model, args, rm_cache=True)

