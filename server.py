import flwr as fl

def get_strategy():
    return fl.server.strategy.FedAvg()