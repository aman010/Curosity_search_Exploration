#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 12:06:05 2025

@author: qb
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results/"
EXPERIMENTS = {
    "PPO": "ppo.csv",
    "PPO + Curiosity": "ppo_curiosity.csv",
    "PPO + Curiosity + Risk": "ppo_curiosity_risk.csv",
    "RND": "rnd.csv",
    "RND + Augmentation": "rnd_aug.csv",
    "Multi-Agent PPO": "multiagent.csv",
    "Multi-Agent + Curiosity": "multiagent_curiosity.csv",
    "Multi-Agent + Curiosity + Risk": "multiagent_curiosity_risk.csv"
}


def load_results():
    data = {}
    for name, file in EXPERIMENTS.items():
        path = os.path.join(RESULTS_DIR, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            data[name] = df
            print(f"[Loaded] {name}")
        else:
            print(f"[Missing] {name}")
    return data



def summarize(df):
    """Extract simple, reproducible metrics."""
    return {
        "mean_reward": df["reward"].mean(),
        "std_reward": df["reward"].std(),
        "max_reward": df["reward"].max(),
        "min_reward": df["reward"].min(),
        "avg_curiosity": df["curiosity"].mean() if "curiosity" in df else np.nan,
        "avg_risk": df["risk_ms"].mean() if "risk_ms" in df else np.nan,
    }




def consolidate(data):
    consolidated = {}
    for name, df in data.items():
        consolidated[name] = summarize(df)
    return pd.DataFrame(consolidated).T


def plot_metrics(data):
    plt.figure(figsize=(10, 6))

    for name, df in data.items():
        plt.plot(df["reward"].rolling(50).mean(), label=name)

    plt.title("Rolling Average Rewards Across Experiments")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("consolidated_reward_plot.png")
    plt.show()




if __name__ == "__main__":
    print("=== CONSOLIDATION SCRIPT RUNNING ===")

    data = load_results()
    table = consolidate(data)

    print("\n=== Consolidated Summary Table ===\n")
    print(table)

    table.to_csv("consolidated_summary.csv")

    print("\n=== Plotting Curves ===")
    plot_metrics(data)

    print("\nSaved:")
    print("- consolidated_summary.csv")
    print("- consolidated_reward_plot.png")
