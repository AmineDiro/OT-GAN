import os
import argparse
#from train import train_and_evaluate
from . import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        '-c',
        "--channels",
        type=int,
        default=3,
        metavar="N",
        help="Nb of channels 1 for MNIST  ; 3 for CIFAR , 3 by default",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="Batch size for training and scoring(default: 64)",
    )
    parser.add_argument(
        '-se',
        "--save_epoch",
        type=int,
        default=2,
        metavar="N",
        help="Saving model every N epoch",
    )
    parser.add_argument(
        '-si',
        "--sample_interval",
        type=int,
        default=1,
        metavar="N",
        help="Interval number for sampling image from generator and saving them ",
    )
    parser.add_argument("--score", dest="score", action="store_true")
    parser.add_argument("--no-score", dest="score", action="store_false")

    parser.set_defaults(score=True)
    args = parser.parse_args()
    train.train_and_evaluate(
        channels=args.channels, 
        batch_size=args.batch_size, 
        INCEPTION_SCORE=args.score,
        save_epoch=args.save_epoch,
        sample_interval=args.sample_interval
    )