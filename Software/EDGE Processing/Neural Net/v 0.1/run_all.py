"""
run_all.py

Runs all neural-network experiments sequentially.

1. PPG-only CNN
2. Single-stream PPG + accelerometer CNN
3. Two-branch PPG + accelerometer CNN
"""

import CNN
import accelCNN
import accelCNN_2branch
import split_config


def main():

    train_ids = split_config.train_ids
    validation_ids = split_config.validation_ids
    test_ids = split_config.test_ids

    print("\n========================================")
    print("STARTING PPG-ONLY CNN")
    print("========================================")

    CNN.main(train_ids, validation_ids, test_ids)

    print("\n========================================")
    print("PPG-ONLY CNN COMPLETE")
    print("STARTING SINGLE-STREAM PPG + ACCEL CNN")
    print("========================================")

    accelCNN.main(train_ids, validation_ids, test_ids)

    print("\n========================================")
    print("SINGLE-STREAM ACCEL CNN COMPLETE")
    print("STARTING TWO-BRANCH PPG + ACCEL CNN")
    print("========================================")

    accelCNN_2branch.main(train_ids, validation_ids, test_ids)

    print("\n========================================")
    print("ALL RUNS COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()