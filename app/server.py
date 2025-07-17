import flwr as fl

def run_server_flwr() -> None:
    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=lambda rnd: {"round": rnd},
        on_evaluate_config_fn=lambda rnd: {"round": rnd},
    )

    print("[SERVER] Iniciando servidor federado...")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

__all__ = ["run_server_flwr"]