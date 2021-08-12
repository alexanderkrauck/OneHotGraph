import utils
import utils.training as training

search_grid = {
    "hidden_channels": [64, 256, 1028],
    "head_depth": [1,2,3,4],
    "base_depth": [3,5,10],
    "base_dropout": [0.5, 0.2],
    "head_dropout": [0.5, 0.2],
    "lr": [1e-2, 1e-3]
}

def main():
    training.search_configs(search_grid, 10)

if __name__ == '__main__':
    main()