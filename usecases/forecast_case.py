import torch
from models.factory import build_model
from training.train_loop import train
from training.losses import get_loss
from training.metrics import get_metrics
from training.optimizer import get_optimizer
from data.data_connecter import get_binance
from utils.logger import Logger
from experiments.eval import evaluate, plot_multistep_predictions

class DLUseCase:
    """
    Deep Learning use case for time series forecasting.

    This class encapsulates the training and evaluation pipeline
    for a supervised model (e.g., RNN, Transformer) applied to
    time series data.
    """

    def __init__(self, config):
        """
        Initialize the DL use case with configuration.

        Args:
            config (Namespace or dict): Configuration object containing all hyperparameters.
        """
        self.config = config
        self.device = torch.device("cpu" if getattr(config, "force_cpu", False) or not torch.cuda.is_available() else "cuda")

        self.model = None
        self.loss_fn = None
        self.metric_fn = None
        self.optimizer = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _setup(self):
        """
        Prepare data loaders, model, loss function, metric and optimizer.
        This is called before training or testing.
        """
        print(f"Using device: {self.device}")
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = get_binance(config=self.config)
        
        # Build model and training components
        self.model = build_model(self.config).to(self.device)
        self.loss_fn = get_loss(self.config.loss)
        self.metric_fn = get_metrics(self.config.metric)
        self.optimizer = get_optimizer(self.model, self.config)

    def train(self):
        """
        Train the model using the training dataset and evaluate periodically on validation set.
        Logs results using the Logger utility.
        """
        self._setup()
        logger = Logger(log_dir="logs", run_name="rnn_experiment")

        train(
            model=self.model,
            dataloader=self.train_loader,
            config=self.config,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            device=self.device,
            logger=logger,
            val_loader=self.val_loader
        )

    def test(self):
        """
        Load the model checkpoint and evaluate it on the test set.
        Plots sample predictions after evaluation.
        """
        self._setup()

        # Load trained model weights
        checkpoint_path = getattr(self.config, "checkpoint_path", "checkpoints/rnn_best.pt")
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(self.device)

        # Evaluate on test data
        test_loss, test_metric, y_pred, y_true = evaluate(
            model=self.model,
            dataloader=self.test_loader,
            config=self.config,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            device=self.device
        )

        plot_multistep_predictions(y_true=y_true, y_pred=y_pred, n_samples=5)
