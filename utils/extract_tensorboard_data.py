import argparse
from tensorboard.backend.event_processing import event_accumulator


def extract_training_logs(log_dir):
    """
    Extract training logs from tensorboard data.

    Args:
        log_dir (str): The directory where tensorboard data is saved.

    Returns:
        dict: A dictionary containing training logs.
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    scalars = ea.Tags()["scalars"]

    # load tensorboard data
    logs = {}
    for scalar in scalars:
        if scalar not in logs:
            logs[scalar] = []

        for event in ea.Scalars(scalar):
            logs[scalar].append((event.step, event.value))

    return logs


def get_values_from_events(events):
    return [event[1] for event in events]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract training logs from tensorboard data."
    )

    parser.add_argument("--log_dir", type=str, help="The tensorboard log directory.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logs = extract_training_logs(args.log_dir)

    print(f"Extracted logs from {args.log_dir}:")
    print(f"Keys: {logs.keys()}")
