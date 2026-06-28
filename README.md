## Quick start

### 1. Clone and install

```bash
git clone https://github.com/mohammedsalmankhan/q-learning-gridworld.git
cd q-learning-gridworld
pip install -r requirements.txt
```

### 2. Run training and reproduce the figures

```bash
python src/main.py
```

This trains a Q-learning agent with alpha=0.5, runs the SARSA comparison, performs the learning rate sweep, and saves all figures to `./plots/`.

### 3. Run the interactive demo

```bash
streamlit run streamlit_app.py
```

A browser tab opens where you can adjust alpha, gamma, epsilon-decay, episodes, and the random seed, and watch the agent learn live.

### 4. Run the tests

```bash
pytest tests/ -v
```

All 14 tests pass in under three seconds.

---

## Usage examples

Default training run:

```bash
python src/main.py
```

Sweep with a different learning rate and seed:

```bash
python src/main.py --alpha 0.3 --seed 7
```

Quick run without the multi-seed sweep:

```bash
python src/main.py --skip-sweep
```

Programmatic use:

```python
from src.environment import GridWorld
from src.agent import QLearningAgent
from src.main import train_q, evaluate_greedy

env = GridWorld()
agent = QLearningAgent(alpha=0.5, seed=42)
rewards, rolling, _, converged = train_q(env, agent)
path, total = evaluate_greedy(GridWorld(), agent)
print(f"Reached goal in {len(path) - 1} steps with reward {total}")
```

---

## Key results

### Learning rate sweep

| alpha | Mean episodes to converge | Mean final reward |
|---|---|---|
| 0.1 | 62.0 | 10.33 |
| 0.3 | 69.7 | 11.00 |
| 0.5 | 59.3 | 11.00 |
| 0.7 | 73.0 | 9.33 |
| 1.0 | 61.3 | 11.00 |

alpha = 0.5 gives the best balance of stability and convergence speed. Higher learning rates converge faster but get noisier once the policy is near-optimal, since each large TD update can destabilise cells that were already well estimated.

![Alpha comparison](docs/plots/alpha_comparison.png)

### Learned state values vs Bellman optimum

The learned values match the Bellman optimum exactly on every cell along the agent's preferred trajectory. Off-trajectory cells keep some residual TD error, which is a known property of model-free Q-learning under finite epsilon-decay: states that are rarely visited don't get enough updates to fully converge.

| Learned V(s) | Optimal V*(s) |
|---|---|
| ![Learned](docs/plots/state_values_learned.png) | ![Optimal](docs/plots/state_values_optimal.png) |

### Greedy policy

![Policy](docs/plots/policy.png)

### Q-learning vs SARSA

Both algorithms converge to the same optimal trajectory here. On this deterministic, hazard-free grid the off-policy and on-policy targets coincide along the trajectory, so the two curves track closely. In a stochastic or "cliff-walking" environment the gap between them would widen considerably.

![Q vs SARSA](docs/plots/q_vs_sarsa.png)

---

## Future improvements

- Function approximation with a small neural network (DQN) to scale beyond tabular state spaces
- Stochastic transitions and slippery cells, to widen the gap between Q-learning and SARSA
- Reward shaping and intrinsic motivation to address the off-trajectory coverage problem
- Larger, procedurally generated grid worlds with curriculum learning
- Persistent best-policy storage on the deployed demo

---

## License

MIT, see [LICENSE](LICENSE).

---

## About

Built by Mohammed Salman Khan.

Email: mohammedsalmankhans636@gmail.com
LinkedIn: https://www.linkedin.com/in/mohammedsalmankhans/
GitHub: https://github.com/mohammedsalmankhan
