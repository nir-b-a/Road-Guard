import csv
import math

def evaluate_estimation(real_csv='real_distance.csv', estimated_csv='distances.csv'):
    real = {}
    est = {}

    # load real distances
    with open(real_csv, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            frame = int(row[0])
            real[frame] = float(row[1])

    # load estimated distances
    with open(estimated_csv, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            frame = int(row[0])
            est[frame] = float(row[1])

    # align frames
    frames = sorted(set(real.keys()) & set(est.keys()))

    errors = []
    abs_errors = []
    sq_errors = []

    for f in frames:
        e = est[f] - real[f]
        errors.append(e)
        abs_errors.append(abs(e))
        sq_errors.append(e * e)

    n = len(frames)

    mae = sum(abs_errors) / n
    mse = sum(sq_errors) / n
    rmse = math.sqrt(mse)

    # mean bias (over/under estimation)
    bias = sum(errors) / n

    print(f"Frames compared: {n}")
    print(f"MAE  (mean abs error): {mae:.4f}")
    print(f"MSE  (mean squared error): {mse:.4f}")
    print(f"RMSE (root MSE): {rmse:.4f}")
    print(f"Bias (mean error): {bias:.4f}")


if __name__ == "__main__":
    evaluate_estimation()