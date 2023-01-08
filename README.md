# DL_Project
Final Project For DL.



## Dataset

WillowObject, which will be automatic downloaded. View `dataset/dataloader.py` for implement details.

-   For training, please set `Train = True, Eval = False`
-   For evaluation, please set `Train = False, Eval = True`
-   For plotting, please set `Train = False, Eval = False`



## Train

Run this to train a PCA-GM model without any pre-train model.

```bash
python train.py --schedular --Shuffle --train_with_eval
```

Run this to train a PCA-GM model with pre-train model

```bash
python train.py --schedular --Shuffle --train_with_eval --use_pretrained_model --pretrained_model_path "..."
```



## Evaluation

It's more recommend to use `--train_with_eval` flag to evalution during the training, but you can still run this to evaluation a specific model:

```bash
python eval.py --path "..." --Eval_model_name "..."
```



## Plot

The Loss curve will automatically be plotted during training. Run this to plot some sample of matching results.

```bash
python plot.py --path "..." --Eval_model_name "..." --plot_saving_path "..."
```
