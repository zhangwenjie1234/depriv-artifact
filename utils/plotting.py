import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def _ema_smooth(values, alpha):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for idx in range(1, len(values)):
        smoothed[idx] = alpha * values[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def plot_rl_training_curves(args, trainer, output_root):
    if not getattr(args, "rl", False):
        return
    if not bool(getattr(args, "plot_rl_curves", 1)):
        return

    agent_type = "rl"
    plots_dir = os.path.join(output_root, "reward_plots")
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if hasattr(trainer, "all_rewards_log") and trainer.all_rewards_log:
        rewards = np.array(trainer.all_rewards_log, dtype=np.float64)
        min_reward_points = 300
        if len(rewards) < min_reward_points:
            print(f"Skipping reward plot: only {len(rewards)} points (< {min_reward_points}).")
        else:
            print("Plotting reward trend...")
            plt.switch_backend("Agg")
            plot_filename = os.path.join(plots_dir, f"{agent_type}_{args.dataset}_{timestamp}.png")
            plot_title = f"{agent_type.upper()} Agent Reward Trend ({args.dataset})"
            smooth_alpha = 0.08 if len(rewards) >= 1000 else 0.12
            smooth_rewards = _ema_smooth(rewards, alpha=smooth_alpha)

            plt.figure(figsize=(12, 6))
            plt.plot(rewards, label="Per-step Reward", alpha=0.12, color="royalblue", linewidth=1.0)
            plt.plot(smooth_rewards, label=f"EMA Trend (alpha={smooth_alpha:.2f})", color="red", linewidth=2.2)
            plt.title(plot_title)
            plt.xlabel("Training Step (Batch)")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(True, alpha=0.35)
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close()
            print(f"Reward plot saved to {plot_filename}")

    if hasattr(trainer, "eval_reward_history") and len(trainer.eval_reward_history) > 0:
        eval_rewards = np.array(trainer.eval_reward_history, dtype=np.float64)
        min_eval_points = 8
        if len(eval_rewards) < min_eval_points:
            print(f"Skipping deterministic reward plot: only {len(eval_rewards)} points (< {min_eval_points}).")
        else:
            print("Plotting Deterministic Reward trend...")
            plt.switch_backend("Agg")
            eval_plot_filename = os.path.join(plots_dir, f"{agent_type}_eval_{args.dataset}_{timestamp}.png")
            eval_x = np.arange(len(eval_rewards))
            eval_smooth = _ema_smooth(eval_rewards, alpha=0.25)

            plt.figure(figsize=(12, 6))
            plt.scatter(eval_x, eval_rewards, label="Eval Reward", color="forestgreen", s=28, alpha=0.75)
            plt.plot(eval_x, eval_smooth, label="EMA Trend", color="darkgreen", linewidth=2.2)
            plt.title(f"Deterministic Policy Performance ({args.dataset})")
            plt.xlabel("Evaluation Points (every 100 steps)")
            plt.ylabel("Average Reward")
            plt.legend()
            plt.grid(True, alpha=0.35)
            plt.tight_layout()
            plt.savefig(eval_plot_filename)
            plt.close()
            print(f"Deterministic Reward plot saved to {eval_plot_filename}")
