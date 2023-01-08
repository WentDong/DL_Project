from argparse import ArgumentParser

def get_args(Title):
    parser = ArgumentParser(Title)

    parser.add_argument(
        "--n_epochs", default=16, type=int, help="the number of epochs to run."
    )
    parser.add_argument(
        "--lr", default = 1e-4, type = float, help="learning rate."
    )
    parser.add_argument(
        "--path", default = "./out", type = str, help="path where the trained model saved, and also where the model under evaluation stored."
    )
    parser.add_argument(
        "--Eval_model_name", default = "pca_gm_15.pkl", type = str, help = "The model's name under evaluation or used to plot."
    )
    parser.add_argument(
        "--save_frequency", default = 3, type = int, help = "The frequency to save a model."
    )
    parser.add_argument(
        "--batch_size", default=4, type = int, help = "Batch Size."
    )
    parser.add_argument(
        "--Shuffle", default=False, action= "store_true", help = "whether shuffle in training"
    )
    parser.add_argument(
        "--train_with_eval", default = False, action="store_true", help = "whether train with eval"
    )
    parser.add_argument(
        "--use_pretrained_model", default = False, action = "store_true", help = "whether use pretrained model"
    )
    parser.add_argument(
        "--pretrained_model_path", default = "./pygmtools-pretrain-models/pca_gm_willow_jittor.pt", type = str, help = "the path of pretrained model"
    )
    parser.add_argument(
        "--schedular", default = False, action = "store_true", help = " Whether use schedular."
    )
    parser.add_argument(
        "--schedular_step", default = 1, type = int, help = "Step of StepLR Schedular"
    )
    parser.add_argument(
        "--schedular_gamma", default= 0.3, type=float, help = "Gamma of StepLR Schedular"
    )
    parser.add_argument(
        "--use_vgg_pretrained_model", default = False, action = "store_true", help = "Whether use vgg_pretrained_model"
    )
    parser.add_argument(
        "--vgg_pretrained_model", default = "./pygmtools-pretrain-models/vgg16_pca_voc_jittor.pt", type = str, help = "The model of pretrained VGG"
    )
    parser.add_argument(
        "--plot_saving_path", default = "./img", type = str, help = "path where the plots will be stored"
    )
    return parser.parse_args()
