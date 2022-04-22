from ast import arg
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
import encoder
import argparse
from trainer import Trainer
from evaluator import Evaluator
from reconstructor import Reconstructor

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005         # learning rate
N_TEST_IMG = 5
MODEL_PATH = './model/encoder.pt'
Trained = True
DATASET = 'mnist'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--test", action='store_true', default=False)
    parser.add_argument("-r", "--reconstruct", action='store_true', default=False)

    args = parser.parse_args()

    autoencoder = encoder.AutoEncoder(DATASET)
    if args.test:
        autoencoder.load_state_dict(torch.load(MODEL_PATH))
        evaluator = Evaluator(autoencoder, BATCH_SIZE, EPOCH, DATASET)
        evaluator.evaluate()
    elif args.reconstruct:
        autoencoder.load_state_dict(torch.load(MODEL_PATH))
        reconstructor = Reconstructor(autoencoder, BATCH_SIZE, EPOCH, DATASET)
        reconstructor.eval_attack()
    else:
        trainer = Trainer(autoencoder, BATCH_SIZE, EPOCH, MODEL_PATH, DATASET)
        trainer.train()






