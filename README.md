# ELENDIL v2

A multi-agent reinforcement learning environment compatible with PettingZoo for simulating UAVs (Unmanned Aerial Vehicles), UGVs (Unmanned Ground Vehicles), and targets in a grid-based world.

## Features

- **Multi-agent environment**: Supports multiple UAVs, UGVs, and targets
- **PettingZoo compatible**: Works seamlessly with PettingZoo's parallel environment API
- **Flexible scenarios**: Explore and track scenarios
- **Communication modes**: None or complete communication between agents
- **Visualization**: Built-in matplotlib-based rendering
- **Altitude-based FOV**: UAVs have altitude-dependent field of view

## Installation

### From source

```bash
git clone <repository-url>
cd ELENDIL_v2
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from elendil_v2 import elendil_v2

# Create environment
env = elendil_v2(
    render_mode="human",
    num_UGVs=1,
    num_UAVs=1,
    num_targets=1,
    scenario="explore",
    map_type="medium",
    seed=42
)

# Reset environment
observations, infos = env.reset()

# Run simulation
for step in range(100):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    if any(terminations.values()) or any(truncations.values()):
        observations, infos = env.reset()

env.close()
```

## Configuration

### Environment Parameters

- `num_UGVs`: Number of ground vehicles (default: 1)
- `num_UAVs`: Number of aerial vehicles (default: 1)
- `num_targets`: Number of targets (default: 1)
- `scenario`: "explore" or "track" (default: "explore")
- `map_type`: "small" (10x10), "medium" (15x15), or "large" (20x20) (default: "medium")
- `communication_style`: "none" or "complete" (default: "none")
- `step_limit`: Maximum number of steps per episode (default: 500)
- `seed`: Random seed for reproducibility (default: None)
- `render_mode`: "human" for visualization or None (default: None)

## Running Tests

```bash
pytest tests/
```

Or run the original test file:

```bash
python -m pytest tests/test_elendil_v2.py
```

## Project Structure

```
ELENDIL_v2/
├── elendil_v2/          # Main package
│   ├── __init__.py      # Package initialization
│   └── env.py           # Environment implementation
├── tests/               # Test suite
│   ├── __init__.py
│   └── test_elendil_v2.py
├── configs/             # Configuration files
│   └── maps/
│       └── medium_env_obstacles.yaml
├── setup.py             # Setup script
├── pyproject.toml       # Modern Python project configuration
└── README.md            # This file
```

## License

MIT License

