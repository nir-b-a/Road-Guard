import csv
import matplotlib.pyplot as plt

def load_csv(path, x):
    frames = []
    values = []

    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header

        for row in reader:
            frames.append(int(row[0]))
            if x == 1:
                values.append(float(row[1]) + 4.4)
            else:
                values.append(float(row[1]))
            

    return frames, values


def plot_distances(real_csv, est_csv):
    real_frames, real_vals = load_csv(real_csv, 0)
    est_frames, est_vals = load_csv(est_csv, 1)

    plt.figure()

    plt.plot(real_frames, real_vals, label="Real Distance")
    plt.plot(est_frames, est_vals, label="Estimated Distance")

    plt.xlabel("Frame")
    plt.ylabel("Distance")
    plt.title("Real vs Estimated Distance over Time")

    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    plot_distances("real_distance.csv", 'distances.csv')