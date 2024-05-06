import matplotlib.pyplot as plt
from extract_tensorboard_data import extract_training_logs, get_values_from_events


def plot(data, title, y_axis_name, filename, save_path=None):
    """
    Args:
        data (dict): dictionary containing model names as keys and their values
            e.g, data = {
                            "ResNet50_train": [0.1, 0.2, 0.3, 0.4],
                            "ResNet50_valid": [0.1, 0.2, 0.3, 0.4],
                            "VGGNet19_train": [0.2, 0.3, 0.4, 0.5],
                            "VGGNet19_valid": [0.2, 0.3, 0.4, 0.5]
                        }
    """

    for model_name, model_accuracy in data.items():
        plt.plot(model_accuracy, label=model_name)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_axis_name)
    plt.legend()
    # plt.show()

    if save_path:
        plt.savefig(f"{save_path}/{filename}.png")

    plt.close()


def clamp_values(values, min_value=0, max_value=9999):
    return [max(min(value, max_value), min_value) for value in values]


if __name__ == "__main__":
    ########################################################
    # no_tfr_log = extract_training_logs("outputs/without/20240505-1618/logs")
    # tfr_log = extract_training_logs("outputs/without/20240503-1531/logs")
    # print(no_tfr_log.keys())

    # key = "Loss/train"
    # no_tfr_values = get_values_from_events(no_tfr_log[key])
    # tfr_values = get_values_from_events(tfr_log[key])

    # no_tfr_values = clamp_values(no_tfr_values, 0, 1)
    # tfr_values = clamp_values(tfr_values, 0, 1)

    # plot_data = {"w/o (tfr: 0)": no_tfr_values, "w/ (tfr: 1)": tfr_values}

    # plot(
    #     data=plot_data,
    #     title="Losses w/ and w/o Teacher Forcing Ratio",
    #     y_axis_name="Loss",
    #     filename="tfr_loss_plot",
    #     save_path="assets",
    # )

    ########################################################
    # no_tfr_log = extract_training_logs("outputs/without/20240505-1618/logs")
    # tfr_log = extract_training_logs("outputs/without/20240503-1531/logs")
    # key = "PSNR/val"
    # no_tfr_values = get_values_from_events(no_tfr_log[key])
    # tfr_values = get_values_from_events(tfr_log[key])

    # compute average psnr
    # no_tfr_avg_psnr = sum(no_tfr_values) / len(no_tfr_values)
    # tfr_avg_psnr = sum(tfr_values) / len(tfr_values)
    # print(f"no_tfr psnr: {no_tfr_avg_psnr}, tfr psnr: {tfr_avg_psnr}")

    # plot_data = {
    #     "w/o (tfr: 0): avg. PSNR = 30.42": no_tfr_values,
    #     "w/ (tfr: 1): avg. PSNR = 20.81": tfr_values,
    # }

    # plot(
    #     data=plot_data,
    #     title="PSNR w/ and w/o Teacher Forcing Ratio",
    #     y_axis_name="PSNR",
    #     filename="tfr_psnr_plot",
    #     save_path="assets",
    # )

    ########################################################
    # small_fast_train = extract_training_logs("outputs/without/20240505-1618/logs")
    # all_fast_train = extract_training_logs("outputs/without/20240503-1530/logs")
    # print(small_fast_train.keys())

    # key = "Loss/train"
    # small_fast_train = get_values_from_events(small_fast_train[key])
    # all_fast_train = get_values_from_events(all_fast_train[key])

    # small_fast_train = clamp_values(small_fast_train, 0, 10)
    # all_fast_train = clamp_values(all_fast_train, 0, 10)

    # small_fast_train = small_fast_train[: len(all_fast_train)]

    # plot_data = {
    #     "small fast train (10/30)": small_fast_train,
    #     "all fast train (30/30)": all_fast_train,
    # }

    # plot(
    #     data=plot_data,
    #     title="Losses w/ small/all fast train",
    #     y_axis_name="Loss",
    #     filename="small_all_fast_train_loss_plot",
    #     save_path="assets",
    # )

    ########################################################
    # small_fast_train = extract_training_logs("outputs/without/20240505-1618/logs")
    # all_fast_train = extract_training_logs("outputs/without/20240503-1530/logs")
    # print(small_fast_train.keys())

    # key = "PSNR/val"
    # small_fast_train = get_values_from_events(small_fast_train[key])
    # all_fast_train = get_values_from_events(all_fast_train[key])

    # # small_fast_train = clamp_values(small_fast_train, 0, 10)
    # # all_fast_train = clamp_values(all_fast_train, 0, 10)

    # small_fast_train = small_fast_train[: len(all_fast_train)]

    # plot_data = {
    #     "small fast train (10/30)": small_fast_train,
    #     "all fast train (30/30)": all_fast_train,
    # }

    # plot(
    #     data=plot_data,
    #     title="PSNR w/ small/all fast train",
    #     y_axis_name="PSNR",
    #     filename="small_all_fast_train_psnr_plot",
    #     save_path="assets",
    # )

    ########################################################
    mono = extract_training_logs("outputs/monotonic/20240507-0202/logs")
    cycl = extract_training_logs("outputs/cyclical/20240503-1526/logs")
    without = extract_training_logs("outputs/without/20240505-1618/logs")
    print(mono.keys())

    key = "KL Annealing/beta"
    mono = get_values_from_events(mono[key])
    cycl = get_values_from_events(cycl[key])
    without = get_values_from_events(without[key])

    cycl = cycl[: len(cycl)]
    without = without[: len(cycl)]

    plot_data = {
        "without": without,
        "cyclical": cycl,
        "monotonic": mono,
    }

    plot(
        data=plot_data,
        title="Beta w/ different KL annealing strategies",
        y_axis_name="Beta",
        filename="kl_strategy_beta_plot",
        save_path="assets",
    )

    ########################################################
    mono = extract_training_logs("outputs/monotonic/20240507-0202/logs")
    cycl = extract_training_logs("outputs/cyclical/20240503-1526/logs")
    without = extract_training_logs("outputs/without/20240505-1618/logs")
    print(mono.keys())

    key = "Loss/train"
    mono = get_values_from_events(mono[key])
    cycl = get_values_from_events(cycl[key])
    without = get_values_from_events(without[key])

    max_value = 30
    mono = clamp_values(mono, 0, max_value)
    cycl = clamp_values(cycl, 0, max_value)
    without = clamp_values(without, 0, max_value)

    cycl = cycl[: len(cycl)]
    without = without[: len(cycl)]

    plot_data = {
        "without": without,
        "cyclical": cycl,
        "monotonic": mono,
    }

    plot(
        data=plot_data,
        title="Losses w/ different KL annealing strategies",
        y_axis_name="Loss",
        filename="kl_strategy_loss_plot",
        save_path="assets",
    )

    ########################################################
    mono = extract_training_logs("outputs/monotonic/20240507-0202/logs")
    cycl = extract_training_logs("outputs/cyclical/20240503-1526/logs")
    without = extract_training_logs("outputs/without/20240505-1618/logs")
    print(mono.keys())

    key = "PSNR/val"
    mono = get_values_from_events(mono[key])
    cycl = get_values_from_events(cycl[key])
    without = get_values_from_events(without[key])

    without = without[: len(cycl)]

    plot_data = {
        "without": without,
        "cyclical": cycl,
        "monotonic": mono,
    }

    plot(
        data=plot_data,
        title="PSNR w/ different KL annealing strategies",
        y_axis_name="PSNR",
        filename="kl_strategy_psnr_plot",
        save_path="assets",
    )
