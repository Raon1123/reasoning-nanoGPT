import sys
from src.config import load_config
from src.train import train

def main():
    """
    Entry point for the training script.
    """
    # Load configuration
    config = load_config(cli_args=sys.argv[1:])
    
    # Start training
    print("Starting training with config:")
    # print(config) # Optional: print config for debugging
    
    train(config)

if __name__ == "__main__":
    main()
