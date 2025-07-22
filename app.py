import flwr as fl
from client import client_fn
from server import get_strategy

def main() -> None:
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.simulation.SimulationConfig(num_rounds=5),
        strategy=get_strategy(),
    )

if __name__ == "__main__":
    main()