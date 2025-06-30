import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from rl.util import *

def evaluate(model, dataloader, config, loss_fn, metric_fn, device):
    """
    Evaluate a trained model on a given dataloader (test or validation set).
    
    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for evaluation.
        config (Namespace): Configuration object (e.g., for verbosity, output_dim).
        loss_fn (callable): Loss function.
        metric_fn (callable): Evaluation metric.
        device (torch.device): CPU or CUDA.

    Returns:
        Tuple (avg_loss, avg_metric, all_preds, all_targets)
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_metric = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss = loss_fn(output, y)
            metric = metric_fn(output, y)

            total_loss += loss.item()
            total_metric += metric

            all_preds.append(output.cpu())
            all_targets.append(y.cpu())

    avg_loss = total_loss / len(dataloader)
    avg_metric = total_metric / len(dataloader)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    print(f"[Evaluation] Loss: {avg_loss:.4f} | Metric: {avg_metric:.4f}")
    return avg_loss, avg_metric, all_preds, all_targets

def plot_multistep_predictions(y_true, y_pred, n_samples=5):
    """
    Visualise multi-step time series predictions vs true values.

    Args:
        y_true (Tensor): Ground truth of shape [N, H].
        y_pred (Tensor): Predictions of shape [N, H].
        n_samples (int): Number of prediction windows to plot.
    """
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    
    for i in range(min(n_samples, len(y_true))):
        plt.figure(figsize=(8, 3))
        plt.plot(range(len(y_true[i])), y_true[i], label='True', marker='o')
        plt.plot(range(len(y_pred[i])), y_pred[i], label='Predicted', marker='x')
        plt.title(f"Sample {i+1}")
        plt.xlabel("Forecast step (e.g. day)")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

def evaluate_rl(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))

