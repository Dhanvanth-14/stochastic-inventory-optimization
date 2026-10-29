"""
Stochastic Inventory Optimization Under Uncertain Demand
Reproducible model using discrete demand scenarios and Monte Carlo simulation.
Run: python stochastic_inventory.py
Requirements: numpy, pandas, matplotlib
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

SELL_PRICE = 100
PURCHASE_COST = 60
SALVAGE_VALUE = 10
DEMAND_VALUES = np.array([20, 40, 60])
DEMAND_PROBS = np.array([0.20, 0.50, 0.30])
CANDIDATE_QUANTITIES = np.arange(0, 81, 5)
N_SIMULATIONS = 100_000
RANDOM_SEED = 42

def calculate_profit(quantity, demand):
    """Profit = sales revenue + salvage revenue - purchasing cost."""
    sold = np.minimum(quantity, demand)
    unsold = np.maximum(quantity - demand, 0)
    return SELL_PRICE * sold + SALVAGE_VALUE * unsold - PURCHASE_COST * quantity

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    simulated_demand = rng.choice(
        DEMAND_VALUES, size=N_SIMULATIONS, p=DEMAND_PROBS
    )

    rows = []
    for q in CANDIDATE_QUANTITIES:
        scenario_profit = calculate_profit(q, DEMAND_VALUES)
        exact_expected_profit = float(np.dot(scenario_profit, DEMAND_PROBS))
        simulated_profit = calculate_profit(q, simulated_demand)
        rows.append({
            "stock_quantity": int(q),
            "profit_demand_20": float(scenario_profit[0]),
            "profit_demand_40": float(scenario_profit[1]),
            "profit_demand_60": float(scenario_profit[2]),
            "expected_profit_inr": exact_expected_profit,
            "simulated_expected_profit_inr": float(np.mean(simulated_profit)),
            "simulated_profit_std_inr": float(np.std(simulated_profit, ddof=1)),
            "simulated_profit_p05_inr": float(np.quantile(simulated_profit, 0.05)),
            "simulated_profit_p95_inr": float(np.quantile(simulated_profit, 0.95)),
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "stock_quantity_results.csv", index=False)

    best = results.loc[results["expected_profit_inr"].idxmax()]
    mean_demand = float(np.dot(DEMAND_VALUES, DEMAND_PROBS))
    deterministic_quantity = int(round(mean_demand / 5) * 5)
    baseline = results.loc[results["stock_quantity"] == deterministic_quantity].iloc[0]

    print("STOCHASTIC INVENTORY OPTIMIZATION RESULTS")
    print(f"Expected demand: {mean_demand:.1f} units")
    print(f"Best stock quantity: {int(best['stock_quantity'])} units")
    print(f"Maximum expected profit: Rs. {best['expected_profit_inr']:.2f}")
    print(f"Deterministic baseline quantity: {deterministic_quantity} units")
    print(f"Baseline expected profit under uncertainty: Rs. {baseline['expected_profit_inr']:.2f}")
    print(f"Monte Carlo estimate at optimum: Rs. {best['simulated_expected_profit_inr']:.2f}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(results["stock_quantity"], results["expected_profit_inr"], marker="o", label="Exact expected profit")
    plt.plot(results["stock_quantity"], results["simulated_expected_profit_inr"], linestyle="--", marker=".", label="Monte Carlo estimate")
    plt.axvline(int(best["stock_quantity"]), linestyle=":", label="Optimal quantity")
    plt.xlabel("Stock quantity (units)")
    plt.ylabel("Expected profit (Rs.)")
    plt.title("Expected Profit by Stock Quantity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "expected_profit.png", dpi=180)
    plt.close()

    sensitivity_rows = []
    for p_high in [0.10, 0.20, 0.30, 0.40, 0.50]:
        probs = np.array([0.50 - p_high, 0.50, p_high])
        expected = [
            float(np.dot(calculate_profit(q, DEMAND_VALUES), probs))
            for q in CANDIDATE_QUANTITIES
        ]
        i = int(np.argmax(expected))
        sensitivity_rows.append({
            "prob_demand_20": probs[0],
            "prob_demand_40": probs[1],
            "prob_demand_60": probs[2],
            "optimal_stock_quantity": int(CANDIDATE_QUANTITIES[i]),
            "maximum_expected_profit_inr": expected[i],
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUT / "sensitivity_analysis.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.plot(sensitivity["prob_demand_60"], sensitivity["optimal_stock_quantity"], marker="o")
    plt.xlabel("Probability of high demand (60 units)")
    plt.ylabel("Optimal stock quantity (units)")
    plt.title("Sensitivity to High-Demand Probability")
    plt.tight_layout()
    plt.savefig(OUT / "sensitivity.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    main()
