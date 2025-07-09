import os
import torch
from tqdm import tqdm
from copy import deepcopy

from rl.util import *

# plt emporarly to test logs. Later w'll implement vizualization modul
import matplotlib.pyplot as plt

def save_model(model, config, suffix="best"):
    os.makedirs("checkpoints", exist_ok=True)
    path = os.path.join("checkpoints", f"{config.model}_{suffix}.pt")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def train(model, dataloader, config, loss_fn, metric_fn, optimizer, device, logger=None, val_loader=None):
    model.to(device)
    model.train()

    best_metric = float('inf')
    stop_counter = 0
    patience = config.patience

    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(config.epochs):
        total_loss = 0.0
        total_metric = 0.0

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_metric += metric_fn(output, y)
        
        avg_loss = total_loss / len(dataloader)
        avg_metric = total_metric / len(dataloader)

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Metric: {avg_metric:.4f}")

        # If validation set
        model.eval()
        total_val_loss = 0.0
        total_val_metric = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_output = model(x_val)
                total_val_loss += loss_fn(val_output, y_val).item()
                total_val_metric += metric_fn(val_output, y_val)

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_metric = total_val_metric / len(val_loader)

        # To test logs
        train_loss_history.append(avg_loss)
        val_loss_history.append(avg_val_loss)

        # If logs enabled
        if logger:
            logger.log_scalar("Loss/train", avg_loss, step=epoch)
            logger.log_scalar("Metric/train", avg_metric, step=epoch)
            logger.log_scalar("Loss/validation", avg_val_loss, step=epoch)
            logger.log_scalar("Metric/validation", avg_val_metric, step=epoch)

            if epoch % 10 == 0:
                fig, ax = plt.subplots()
                ax.plot(train_loss_history, label="Train Loss")
                ax.plot(val_loss_history, label="Validation Loss")
                ax.set_title("Loss over Epochs")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                logger.log_figure("Loss Curves", fig, step=epoch, prefix="figures")
                plt.close(fig)

        # Manual early stopping (w'll be depreciated to use validation instead)
        if avg_metric < best_metric:
            best_metric = avg_metric
            stop_counter = 0
            save_model(model, config, suffix="best")
        else:
            stop_counter += 1
            print(f"No improvement for {stop_counter} epoch(s)")

        if stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

def train_rl(num_iterations, agent, env,  evaluate, validate_steps, output, config, max_episode_length=None, debug=False):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= config.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        
        # env response with next_observation, reward, terminate_info
        obs, reward, terminated, truncated, info = env.step(action)
        obs = deepcopy(obs)
        if max_episode_length and episode_steps >= max_episode_length -1:
            terminated = True

        # agent observe and update policy
        agent.observe(reward, obs, terminated)
        if step > config.warmup :
            agent.update_policy()
        
        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(obs)

        if terminated: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1