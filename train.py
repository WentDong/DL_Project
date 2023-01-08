'''
This is the train program.
Please check args.py to view the controllable arguments.
You can use train_with_eval to evaluation after saving.
'''
import pygmtools as pygm
import jittor as jt
from configs import *
import numpy as np
from models.model import GMNET
from dataset.dataloader import WillowObject
import os
from tqdm import *
from matplotlib import pyplot as plt
pygm.BACKEND = 'jittor'
from args import *
from eval import evaluation

if __name__ == "__main__":
    args = get_args("PCA_GM")
    jt.flags.use_cuda = jt.has_cuda
    if (args.use_vgg_pretrained_model):
        model = GMNET(args.vgg_pretrained_model)
    if (args.use_pretrained_model):
        model.load_state_dict(jt.load(args.pretrained_model_path))

    TrainLoader = WillowObject(Train = True, Shuffle = args.Shuffle).set_attrs(batch_size = args.batch_size, shuffle = True)
    if (args.train_with_eval):
        TestLoader =  WillowObject(Train=False, Shuffle=False, Eval=True).set_attrs(batch_size=args.batch_size,
                                                                               shuffle=False)
    optim = jt.optim.Adam(model.parameters(),lr = args.lr)

    if args.schedular:
        scheduler = jt.lr_scheduler.StepLR(optim, args.schedular_step, args.schedular_gamma)

    if not os.path.exists(args.path):
        os.mkdir(args.path)
    losses = []
    losses_idx = []

    for epoch in range(args.n_epochs):
        loop = tqdm(enumerate(TrainLoader), total=len(TrainLoader)//args.batch_size)
        lens = len(TrainLoader)//args.batch_size
        for index, pairs in loop:
            if (args.Shuffle):
                img1, img2, kpts1, kpts2, A1, A2, idx = pairs
                X = model(img1, img2, kpts1, kpts2, A1, A2)
                X_gt = jt.stack([jt.init.eye(X.shape[1])[:,idx[i]] for i in range(X.shape[0])])
            else:
                img1, img2, kpts1, kpts2, A1, A2 = pairs
                X = model(img1, img2, kpts1, kpts2, A1, A2)
                X_gt = jt.stack([jt.init.eye(X.shape[1]) for _ in range(X.shape[0])])

            loss = pygm.utils.permutation_loss(X, X_gt)

            optim.backward(loss)
            optim.step()
            optim.zero_grad()
            loop.set_description(f'Epoch [{epoch}/{args.n_epochs}]')
            loop.set_postfix(loss = loss)

            losses.append(loss.numpy()[0])
            losses_idx.append(epoch * lens + index)

        if (args.schedular):
            scheduler.step()

        if (epoch % args.save_frequency == 0):
            print(f"Epoch: {epoch}. Save model.")
            model.save(os.path.join(args.path, f"pca_gm_{epoch}.pkl"))

            if (args.train_with_eval):
                print("Start Evalutation.")
                evaluation(TestLoader, model, args, False)

        plt.plot(losses_idx, losses)
        plt.savefig(os.path.join(args.path, "loss.png"))

    np.save(os.path.join(args.path, "loss_idx.npy"), losses_idx)
    np.save(os.path.join(args.path, "loss.npy"), losses)
    print("Finish Training and Start Saving Model.")
    model.save(os.path.join(args.path, f"pca_gm_{args.n_epochs-1}.pkl"))
    
    if args.train_with_eval:
        print("Start Evaluation.")
        evaluation(TestLoader, model, args, True)