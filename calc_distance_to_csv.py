import csv
import math

def compute_distance_csv(input_path, output_path):
    with open(input_path, "r", newline="") as f_in, open(output_path, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # write header
        writer.writerow(["frame", "distance"])

        header = next(reader, None)  # skip header if exists

        for row in reader:
            if not row:
                continue

            frame = int(row[0])

            ego_x = float(row[2])
            ego_y = float(row[3])
            ego_z = float(row[4])

            target_x = float(row[5])
            target_y = float(row[6])
            target_z = float(row[7])

            dx = ego_x - target_x
            dy = ego_y - target_y
            dz = ego_z - target_z

            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            writer.writerow([frame, distance])


if __name__ == "__main__":
    compute_distance_csv('telemetry_0.csv', 'real_distance.csv')