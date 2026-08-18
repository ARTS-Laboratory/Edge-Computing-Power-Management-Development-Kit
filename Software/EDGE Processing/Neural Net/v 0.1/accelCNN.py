"""
accelCNN.py

Version: 0.1a

1D CNN using raw PATCH PPG + XYZ accelerometer.

Training:
8 PA recordings

Validation:
2 PA recordings

Testing:
2 PA recordings
"""

import copy
import csv
import numpy as np
import torch
import torch.nn as nn

from import_data_accel import prepare_dataset
from torch.utils.data import TensorDataset, DataLoader


class PPGCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),

            nn.Linear(32 * 187, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def main(train_ids, validation_ids, test_ids):

    X_train, y_train, X_validation, y_validation, X_test, y_test, motion_train, motion_validation, motion_test = prepare_dataset("Data", train_ids, validation_ids, test_ids, plot=False)

    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_validation.shape, y_validation.shape)
    print("Testing:", X_test.shape, y_test.shape)

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_validation = torch.tensor(X_validation, dtype=torch.float32).permute(0, 2, 1)
    y_validation = torch.tensor(y_validation, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    motion_test = torch.tensor(motion_test, dtype=torch.float32)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_validation:", X_validation.shape)
    print("y_validation:", y_validation.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    baseline = y_train.mean()
    baseline_validation_mae = torch.mean(torch.abs(y_validation - baseline))
    baseline_test_mae = torch.mean(torch.abs(y_test - baseline))

    print("\nBaseline HR:", baseline.item())
    print("Baseline validation MAE:", baseline_validation_mae.item(), "BPM")
    print("Baseline test MAE:", baseline_test_mae.item(), "BPM")

    low_threshold = torch.quantile(motion_test, 0.33)
    high_threshold = torch.quantile(motion_test, 0.67)

    low_motion = motion_test <= low_threshold
    medium_motion = (motion_test > low_threshold) & (motion_test <= high_threshold)
    high_motion = motion_test > high_threshold

    print("\nMotion thresholds")
    print("Low/Medium:", low_threshold.item())
    print("Medium/High:", high_threshold.item())

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    seeds = [1, 2, 3, 4, 5]
    results = []

    for run, seed in enumerate(seeds, start=1):

        print("\n========================================")
        print("RUN:", run, "SEED:", seed)
        print("========================================")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = PPGCNN()

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        epochs = 100
        patience = 10

        best_validation_mae = float("inf")
        best_epoch = 0
        best_state = None
        patience_counter = 0

        print("Training")

        for epoch in range(epochs):

            model.train()
            total_loss = 0.0

            for X_batch, y_batch in train_loader:

                predictions = model(X_batch).squeeze()
                loss = loss_function(predictions, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(X_batch)

            training_loss = total_loss / len(train_dataset)

            model.eval()

            with torch.no_grad():
                validation_predictions = model(X_validation).squeeze()
                validation_mae = torch.mean(torch.abs(validation_predictions - y_validation)).item()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print("Epoch:", epoch + 1, "Training Loss:", training_loss, "Validation MAE:", validation_mae)

            if validation_mae < best_validation_mae:
                best_validation_mae = validation_mae
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0

            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping at epoch:", epoch + 1)
                break

        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            train_predictions = model(X_train).squeeze()
            validation_predictions = model(X_validation).squeeze()
            test_predictions = model(X_test).squeeze()

            train_errors = train_predictions - y_train
            validation_errors = validation_predictions - y_validation
            test_errors = test_predictions - y_test

            train_mae = torch.mean(torch.abs(train_errors)).item()
            validation_mae = torch.mean(torch.abs(validation_errors)).item()
            test_mae = torch.mean(torch.abs(test_errors)).item()
            test_rmse = torch.sqrt(torch.mean(test_errors ** 2)).item()

            correlation = torch.corrcoef(torch.stack((y_test, test_predictions)))[0, 1].item()
            bias = torch.mean(test_predictions - y_test).item()

            low_motion_mae = torch.mean(torch.abs(test_predictions[low_motion] - y_test[low_motion])).item()
            medium_motion_mae = torch.mean(torch.abs(test_predictions[medium_motion] - y_test[medium_motion])).item()
            high_motion_mae = torch.mean(torch.abs(test_predictions[high_motion] - y_test[high_motion])).item()

        print("\nRun results")
        print("Best epoch:", best_epoch)
        print("Train MAE:", train_mae, "BPM")
        print("Validation MAE:", validation_mae, "BPM")
        print("Test MAE:", test_mae, "BPM")
        print("Test RMSE:", test_rmse, "BPM")
        print("Correlation:", correlation)
        print("Bias:", bias, "BPM")
        print("Low motion MAE:", low_motion_mae, "BPM")
        print("Medium motion MAE:", medium_motion_mae, "BPM")
        print("High motion MAE:", high_motion_mae, "BPM")

        results.append([
            run,
            seed,
            best_epoch,
            train_mae,
            validation_mae,
            test_mae,
            test_rmse,
            correlation,
            bias,
            low_motion_mae,
            medium_motion_mae,
            high_motion_mae
        ])

    results_array = np.asarray([row[2:] for row in results], dtype=np.float32)

    means = results_array.mean(axis=0)
    stds = results_array.std(axis=0, ddof=1)

    metric_names = [
        "Best Epoch",
        "Train MAE",
        "Validation MAE",
        "Test MAE",
        "Test RMSE",
        "Correlation",
        "Bias",
        "Low Motion MAE",
        "Medium Motion MAE",
        "High Motion MAE"
    ]

    print("\n========================================")
    print("5 RUN SUMMARY")
    print("========================================")

    for name, mean, std in zip(metric_names, means, stds):
        print(f"{name}: {mean:.3f} +/- {std:.3f}")

    output_file = "AccelCNN_v0.1a_5run_results.csv"

    with open(output_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Run",
            "Seed",
            "Best Epoch",
            "Train MAE",
            "Validation MAE",
            "Test MAE",
            "Test RMSE",
            "Correlation",
            "Bias",
            "Low Motion MAE",
            "Medium Motion MAE",
            "High Motion MAE"
        ])

        writer.writerows(results)

        writer.writerow([])

        writer.writerow([
            "Mean",
            "",
            *means
        ])

        writer.writerow([
            "Standard Deviation",
            "",
            *stds
        ])

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()