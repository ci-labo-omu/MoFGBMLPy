import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

def read_summary_file(path):
    regex = re.compile(r"^(.+?):\s+([-+]?[0-9]*\.?[0-9]+)") #(.+?) is for non-greedy match

    metrics = {}
    with open(path, "r") as f:
        for line in f:
            m = regex.match(line.strip())
            if m:
                key = m.group(1).strip()
                val = float(m.group(2))
                metrics[key] = val
    return metrics

def collect_data(base_path):
    records = []

    for test_id, test_name in enumerate(os.listdir(base_path)):
        if test_name == "param_search":
            continue
        data_path = os.path.join(base_path, test_name)
        for data_name in os.listdir(data_path):

            test_path = os.path.join(data_path, data_name)
            if not os.path.isdir(test_path):
                continue

            is_param_search = any(
                os.path.isdir(os.path.join(test_path, d)) for d in os.listdir(test_path)
            )

            if is_param_search:
                for run in os.listdir(test_path):
                    run_dir = os.path.join(test_path, run)
                    if not os.path.isdir(run_dir):
                        continue

                    summary_path = os.path.join(run_dir, "results_summary.txt")
                    if not os.path.exists(summary_path):
                        continue

                    try:
                        param_value = float(run.split("_")[-1])
                    except ValueError:
                        continue

                    metrics = read_summary_file(summary_path)
                    record = {
                        "test_name": test_name,  # = param_name
                        "dataset": data_name,
                        "param_value": param_value,
                    }
                    record.update(metrics)

                    records.append(record)
            else:
                summary_path = os.path.join(test_path, "results_summary.txt")
                if not os.path.exists(summary_path):
                    continue

                metrics = read_summary_file(summary_path)
                record = {
                    "test_name": test_name,
                    "dataset": data_name,
                }
                record.update(metrics)
                records.append(record)

    return pd.DataFrame(records)


def get_boxes_data(df, metric_name, group_by_key):
    boxes = []
    for key, item_data in df.groupby(group_by_key):
        stats_dict = {
            "label": str(key),
            "whislo": item_data[f"{metric_name}_min"].values[0],
            "q1": item_data[f"{metric_name}_q1"].values[0],
            "med": item_data[f"{metric_name}_median"].values[0],
            "q3": item_data[f"{metric_name}_q3"].values[0],
            "whishi": item_data[f"{metric_name}_max"].values[0],
            "fliers": []
        }
        boxes.append(stats_dict)

    return boxes

def plot_metrics(df, out_path):
    os.makedirs(out_path, exist_ok=True)
    is_param_search = "param_value" in df.columns

    all_metrics = defaultdict(set)

    required_stats = {"min", "q1", "median", "q3", "max"}

    for c in df.columns:
        if c not in ["test_name", "dataset", "param_value"]:
            if "_" in c:
                name, stat_type = c.rsplit("_", 1)
                all_metrics[name].add(stat_type)

    # raise an exception if not all metrics have required stats
    for name, stats in all_metrics.items():
        if not required_stats.issubset(stats):
            raise ValueError(f"Metric {name} does not have all required stats: {stats}")

    row_keys = sorted(df["dataset"].unique())
    num_rows = len(row_keys)

    col_keys = None
    num_cols = 1

    if is_param_search:
        col_keys = sorted(df["test_name"].unique())
        num_cols = len(col_keys)

    for metric_name, stats in tqdm(all_metrics.items()):
        # we write a file per metric

        fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=(6 * num_cols, 5 * num_rows),
            sharey=False
        )

        for i, dataset in enumerate(row_keys):
            df_plot_row = df[df["dataset"] == dataset]

            min_y_val = df_plot_row[f"{metric_name}_min"].min()
            max_y_val = df_plot_row[f"{metric_name}_max"].max()

            y_range = max_y_val - min_y_val if min_y_val != max_y_val else 0.1
            y_low = min_y_val - 0.05 * y_range
            y_up = max_y_val + 0.05 * y_range

            if col_keys is not None:
                for j, param in enumerate(col_keys):
                    ax = axes[i][j]
                    df_plot_item = df_plot_row[df_plot_row["test_name"] == param]

                    if df_plot_item.empty:
                        ax.set_title(f"No Data")
                        continue

                    boxes = get_boxes_data(df_plot_item, metric_name, "param_value")

                    if len(boxes) > 0:
                        ax.bxp(boxes, showfliers=False)
                        ax.set_ylim(y_low, y_up)
                        ax.set_xticklabels([b["label"] for b in boxes], rotation=45, ha="right")

                    if i == 0:
                        ax.set_title(param)
                    if j == 0:
                        num_runs = int(df_plot_item["Number of runs"].iloc[0]) if not df_plot_item.empty else 0
                        ax.set_ylabel(f"Metric value ({dataset})\n({num_runs} runs)")

                    ax.set_xlabel("Param value")
                    ax.grid(True)
            else:
                boxes = get_boxes_data(df_plot_row, metric_name, "test_name")
                ax = axes[i]

                if len(boxes) > 0:
                    ax.bxp(boxes, showfliers=False)
                    ax.set_xticklabels([b["label"] for b in boxes], rotation=45, ha="right")
                    ax.set_title(f"{dataset}")
                    ax.set_ylabel(metric_name)
                    ax.set_ylim(y_low, y_up)
                    ax.grid(True)
                else:
                    ax.set_title(f"No Data")

        fig.suptitle(metric_name, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95]) # (left, bottom, right, top), 0.95 to have a small margin on top for title

        plot_file_path = os.path.join(out_path, f"{metric_name}.png")
        plt.savefig(plot_file_path)
        plt.close()


def create_subplot(df, x_key, dataset_name, ax, do_set_y_label=True):
    df = df.sort_values(x_key)
    width = 0.6

    if df[x_key].dtype == float or df[x_key].dtype == int:
        width = (df[x_key].max() - df[x_key].min()) / (len(df) * 2)

    ax.bar(
        df[x_key],
        df["success_rate"],
        width=width,
        edgecolor='black'
    )

    ax.set_ylim(-0.05, 1.05)
    if do_set_y_label:
        num_runs = int(df["Number of runs"].iloc[0]) if not df.empty else 0
        ax.set_ylabel(f"Success rate ({dataset_name})\n({num_runs} runs)")
    # ax.grid(True, axis="y")

    return ax


def plot_success_rate_lineplots(df, out_path):
    os.makedirs(out_path, exist_ok=True)

    is_param_search = "param_value" in df.columns

    row_keys = sorted(df["dataset"].unique())
    num_rows = len(row_keys)

    col_keys = None
    num_cols = 1

    if is_param_search:
        col_keys = sorted(df["test_name"].unique())
        num_cols = len(col_keys)

    if "Number of runs" in df.columns and "Number of failures" in df.columns:
        df["success_rate"] = (df["Number of runs"] - df["Number of failures"]) / df["Number of runs"]
    else:
        raise ValueError("Expected columns 'Number of runs' and 'Number of failures' in data")

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(5*num_cols, 6*num_rows),
        sharey=True,
    )

    for i, dataset in enumerate(row_keys):
        df_row = df[df["dataset"] == dataset]

        if col_keys is not None:
            for j, param in enumerate(col_keys):
                df_plot_item = df_row[df_row["test_name"] == param]

                ax = create_subplot(df_plot_item, "param_value", dataset, axes[i][j], j == 0)

                if i == 0:
                    ax.set_title(param)

                ax.set_xlabel("Param value")

        else:
            ax = create_subplot(df_row, "test_name", dataset, axes[i])

            ax.set_xlabel("Test name")
            ax.set_title(f"Tests on {dataset}")
            ax.set_xticks(range(len(df_row["test_name"])))
            ax.set_xticklabels(df_row["test_name"], rotation=45, ha="right")

    fig.suptitle("Success Rate across Params and Datasets", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # (left, bottom, right, top), 0.95 to have a small margin on top for title

    plot_file_path = os.path.join(out_path, "success_rates.png")
    plt.savefig(plot_file_path)
    plt.close()


if __name__ == "__main__":
    data_path = "../cf_results/cf_metaheuristics/param_search/min_num_rules_2"
    out_path = "../cf_results/plots/cf_metaheuristics/param_search"

    # data_path = "../cf_results/cf_metaheuristics"
    # out_path = "../cf_results/plots/cf_metaheuristics/general"

    df = collect_data(data_path)

    plot_metrics(df, out_path)
    plot_success_rate_lineplots(df, out_path)
