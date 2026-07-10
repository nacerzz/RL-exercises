"""
Grid Search Hyperparameter Optimization for PPO

Week 10

Optimizes:
    - Actor learning rate
    - Critic learning rate
    - Batch size

using a simple exhaustive grid search.

Afterwards the best configuration is evaluated on
unseen random seeds to study generalization.
"""

import itertools
import os

import gymnasium as gym
import pandas as pd
from ppo import PPOAgent, set_seed

# ============================================================
# SEARCH SPACE
# ============================================================

LR_ACTOR = [
    1e-4,
    5e-4,
    1e-3,
]

LR_CRITIC = [
    5e-4,
    1e-3,
    2e-3,
]

BATCH_SIZE = [
    32,
    64,
    128,
]

ENV_NAME = "CartPole-v1"

# Generate every possible combination
GRID = list(
    itertools.product(
        LR_ACTOR,
        LR_CRITIC,
        BATCH_SIZE,
    )
)

print(f"Number of configurations: {len(GRID)}")


# ============================================================
# TRAIN ONE CONFIGURATION
# ============================================================


def train_one_configuration(
    lr_actor,
    lr_critic,
    batch_size,
    seed=0,
):
    """
    Train PPO with one hyperparameter configuration.

    Returns
    -------
    float
        Final evaluation return.
    """

    print("=" * 60)
    print(
        f"Seed={seed} | "
        f"lr_actor={lr_actor} | "
        f"lr_critic={lr_critic} | "
        f"batch={batch_size}"
    )

    env = gym.make(ENV_NAME)

    set_seed(env, seed)

    agent = PPOAgent(
        env,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=4,
        batch_size=batch_size,
        ent_coef=0.01,
        vf_coef=0.5,
        seed=seed,
        hidden_size=128,
    )

    final_return = agent.train(
        total_steps=20000,
        eval_interval=10000,
        eval_episodes=5,
    )

    env.close()

    return final_return


def run_grid_search():

    os.makedirs("Results", exist_ok=True)

    results = []

    print("\nStarting Grid Search...")
    print(f"Total configurations: {len(GRID)}")

    for i, (lr_actor, lr_critic, batch_size) in enumerate(GRID):
        print("\n")
        print("=" * 70)
        print(f"Configuration {i + 1}/{len(GRID)}")
        print("=" * 70)

        score = train_one_configuration(
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            batch_size=batch_size,
            seed=0,
        )

        print(f"Final Evaluation Return: {score:.2f}")

        results.append(
            {
                "lr_actor": lr_actor,
                "lr_critic": lr_critic,
                "batch_size": batch_size,
                "seed": 0,
                "return": score,
            }
        )

    df = pd.DataFrame(results)

    csv_path = "Results/hpo_results.csv"

    df.to_csv(
        csv_path,
        index=False,
    )

    print("\nGrid Search Finished.")
    print(f"Results saved to {csv_path}")

    return df


def find_best_configuration(df):

    best = df.loc[df["return"].idxmax()]

    print("\n")
    print("=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)

    print(best)

    return {
        "lr_actor": float(best["lr_actor"]),
        "lr_critic": float(best["lr_critic"]),
        "batch_size": int(best["batch_size"]),
    }


# ============================================================
# GENERALIZATION TEST
# ============================================================


def evaluate_best_configuration(best_config):

    print("\n")
    print("=" * 70)
    print("GENERALIZATION TEST")
    print("=" * 70)

    results = []

    # Test the best hyperparameters on unseen seeds
    for seed in [1, 2, 3, 4]:
        print(f"\nTesting on seed {seed}")

        score = train_one_configuration(
            lr_actor=best_config["lr_actor"],
            lr_critic=best_config["lr_critic"],
            batch_size=best_config["batch_size"],
            seed=seed,
        )

        results.append(
            {
                "seed": seed,
                "return": score,
            }
        )

    df = pd.DataFrame(results)

    csv_path = "Results/generalization_results.csv"

    df.to_csv(csv_path, index=False)

    print("\nGeneralization results saved to:")
    print(csv_path)

    print("\nAverage Return:", df["return"].mean())
    print("Standard Deviation:", df["return"].std())

    return df


# ============================================================
# MAIN
# ============================================================


def main():

    # Run the hyperparameter search
    hpo_results = run_grid_search()

    # Find the best hyperparameter configuration
    best_config = find_best_configuration(hpo_results)

    # Test whether it generalizes to other seeds
    evaluate_best_configuration(best_config)

    # score = train_one_configuration(
    #     lr_actor=5e-4,
    #     lr_critic=1e-3,
    #     batch_size=64,
    #     seed=0,
    # )
    #
    # print("Returned score:", score)
    # print("\n")
    # print("=" * 70)
    # print("HPO FINISHED")
    # print("=" * 70)


if __name__ == "__main__":
    main()
