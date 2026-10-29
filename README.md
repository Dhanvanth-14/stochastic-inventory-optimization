# Stochastic Inventory Optimization Under Uncertain Demand

## Project overview
This project studies a retailer's inventory decision when customer demand is uncertain. It compares candidate stock quantities using a stochastic model, exact expected-profit calculations, and Monte Carlo simulation.

**Main result under the stated assumptions:** the best tested quantity is **40 units**, with an exact expected profit of **₹1240.00**.

## Academic-data note
All prices, demand values, and probabilities in this project are illustrative assumptions created for an academic simulation. They are not real company data and are not a demand forecast.

## Assumptions
- Selling price: ₹100 per unit
- Purchase cost: ₹60 per unit
- Salvage value for each unsold unit: ₹10
- Demand scenarios: 20, 40, 60 units
- Demand probabilities: 20%, 50%, 30%
- Candidate stock quantities: 0 to 80 units, in increments of 5
- Monte Carlo runs: 100,000
- Random seed: 42

## Model
For stock quantity Q and realized demand D:

Profit(Q,D) = selling_price × min(Q,D) + salvage_value × max(Q−D,0) − purchase_cost × Q.

The objective is to maximize expected profit across the demand distribution. The code also compares exact expected profit with a Monte Carlo estimate.

## Files
- `stochastic_inventory_optimization.ipynb` — notebook walkthrough and analysis
- `stochastic_inventory.py` — reproducible Python script
- `report.pdf` — project report
- `outputs/stock_quantity_results.csv` — profit and simulation results
- `outputs/sensitivity_analysis.csv` — sensitivity results
- `outputs/demand_scenarios.csv` — demand distribution
- `outputs/expected_profit.png` — expected profit chart
- `outputs/sensitivity.png` — sensitivity chart
- `outputs/scenario_profit.png` — scenario profit chart

## Run locally
Install dependencies:

```bash
pip install numpy pandas matplotlib jupyter
```

Run:

```bash
python stochastic_inventory.py
```

Or open `stochastic_inventory_optimization.ipynb` in Jupyter Notebook.

## Limitations
The model assumes a single product, known demand probabilities, fixed prices and costs, and no lead-time, storage, or supplier-capacity constraints. The optimal decision is conditional on these assumptions.
