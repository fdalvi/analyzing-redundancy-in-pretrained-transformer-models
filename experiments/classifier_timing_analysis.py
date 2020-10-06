import argparse
import sys
import time

import numpy as np

if 'AUX_CLASSIFIER_PATH' in os.environ:
    sys.path.append(os.environ['AUX_CLASSIFIER_PATH'])
from aux_classifier import utils

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_instances', type=int, default=1000)
    parser.add_argument('--num_features', type=int)
    parser.add_argument('--num_classes', type=int, default=2)

    args = parser.parse_args()

    X = np.random.random((args.num_instances, args.num_features)).astype(np.float32)
    y = np.random.randint(0, args.num_classes, (args.num_instances,))

    start_time = time.time()
    print("Global start time:", start_time)
    model = utils.train_logreg_model(
        X,
        y,
        lambda_l1=0.00001,
        lambda_l2=0.00001,
        num_epochs=10,
        batch_size=128
    )
    end_time = time.time()
    print("Global end time:", end_time)
    print("Total time: %0.4f seconds (Num Instances: %d, Num Features: %d, Num Classes: %d)" % (end_time-start_time, args.num_instances, args.num_features, args.num_classes))

if __name__ == "__main__":
    main()
