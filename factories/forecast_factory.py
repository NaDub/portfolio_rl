from models.factory import build_model
from training.train_loop import train
from training.losses import get_loss
from training.metrics import get_metrics
from training.optimizer import get_optimizer
from data.data_connecter import get_binance
from utils.logger import Logger
from experiments.eval import evaluate, plot_multistep_predictions

import torch

def get_device(force_cpu=False):
    return torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")

def forecast_factory(config, mode='train'):
    # Selecting device
    device = get_device(config.force_cpu)
    print(f"Using device: {device}")

    # Get data 
    train_loader, val_loader, test_loader = get_binance(config=config)

    # Build components
    model = build_model(config).to(device)
    loss_fn = get_loss(config.loss)
    metric_fn = get_metrics(config.metric)
    optimizer = get_optimizer(model, config)

    # Train or test
    if mode == 'train':
        logger = Logger(log_dir="logs", run_name="rnn_experiment")
        train(model=model, dataloader=train_loader, config=config, loss_fn=loss_fn, metric_fn=metric_fn, optimizer=optimizer, device=device, logger=logger, val_loader=val_loader)

    elif mode == 'test':
        # Testing function evaluate (WIP)
        model.load_state_dict(torch.load("checkpoints/rnn_best.pt"))
        model.to(device)
        
        test_loss, test_metric, y_pred, y_true = evaluate(
        model=model,
        dataloader=test_loader,
        config=config,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device
        )

        plot_multistep_predictions(y_true=y_true, y_pred=y_pred, n_samples=5)

    else:
        raise RuntimeError('undefined mode {}'.format(mode))