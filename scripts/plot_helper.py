import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import shutil

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

def collect_data(base_path, exclude_test_names=None, filter_test_names=None, filter_datasets=None, filter_params=None, param_values_range=None, exclude_datasets=None):
    records = []

    for test_id, test_name in enumerate(os.listdir(base_path)):
        if test_name == "param_search":
            continue
        if filter_test_names is not None and test_name not in filter_test_names:
            continue
        if exclude_test_names is not None and test_name in exclude_test_names:
            continue
        data_path = os.path.join(base_path, test_name)
        for data_name in os.listdir(data_path):
            if filter_datasets is not None and data_name not in filter_datasets:
                continue

            if exclude_datasets is not None and data_name in exclude_datasets:
                continue

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
                        if param_values_range is not None:
                            if not (param_values_range[0] <= param_value <= param_values_range[1]):
                                continue
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
    pd_object = pd.DataFrame(records)

    if filter_params is not None:
        if "param_value" not in pd_object.columns:
            columns_to_keep = ["test_name", "dataset", "Number of runs", "Number of failures"]
            for col in pd_object.columns:
                if col not in columns_to_keep:
                    if any(fp in col for fp in filter_params):
                        columns_to_keep.append(col)

            pd_object = pd_object[columns_to_keep]
        else:
            pd_object = pd_object[pd_object["test_name"].isin(filter_params)]

    return pd_object


def get_boxes_data(df, metric_name, group_by_key):
    boxes = []
    for key, item_data in df.groupby(group_by_key):
        label = str(key)
        if group_by_key == "dataset":
            # num_runs = int(item_data["Number of runs"].iloc[0]) if not item_data.empty else 0
            label = f"{label}"  # ({num_runs} runs)"

        stats_dict = {
            "label": label,
            "whislo": item_data[f"{metric_name}_min"].values[0],
            "q1": item_data[f"{metric_name}_q1"].values[0],
            "med": item_data[f"{metric_name}_median"].values[0],
            "q3": item_data[f"{metric_name}_q3"].values[0],
            "whishi": item_data[f"{metric_name}_max"].values[0],
            "fliers": []
        }
        boxes.append(stats_dict)

    return boxes


def get_y_lims(df, metric_name):
    if metric_name == "success_rate" or ("error_rate" in metric_name and "variation" not in metric_name):
        return -0.05, 1.05
    if "error_rate" in metric_name and "variation" in metric_name:
        # return -0.6, 0.6
        return -1.05, 1.05

    if f"{metric_name}_min" in df.columns:
        min_y_val = df[f"{metric_name}_min"].min()
        max_y_val = df[f"{metric_name}_max"].max()
    else:
        min_y_val = df[metric_name].min()
        max_y_val = df[metric_name].max()

    y_range = max_y_val - min_y_val if min_y_val != max_y_val else 0.1
    y_low = min_y_val - 0.05 * y_range
    y_up = max_y_val + 0.05 * y_range

    return y_low, y_up


def get_method_title(method_name):
    if method_name == "cf_metaheuristics":
        return "Evolutionary-based"
    elif method_name == "cf_gradient":
        return "Gradient-based"
    else:
        raise ValueError(f"Unknown method name: {method_name}")


def plot_metrics(df, out_path, method_name, aggregate_datasets=False):
    method_name = get_method_title(method_name)

    if df.empty:
        return
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

        if num_cols == 1 and aggregate_datasets and df.groupby("test_name").ngroups == 1:
            gen_plot_aggregate(df, metric_name, out_path, method_name)
            continue

        fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=(6 * num_cols, 5 * num_rows),
            sharey=False
        )

        if num_rows == 1:
            axes = [axes]
        if col_keys is not None and num_cols == 1:
            axes = [[ax] for ax in axes]

        for i, dataset in enumerate(row_keys):
            df_plot_row = df[df["dataset"] == dataset]

            y_low, y_up = get_y_lims(df_plot_row, metric_name)

            if col_keys is not None:
                for j, param in enumerate(col_keys):
                    ax = axes[i][j]
                    df_plot_item = df_plot_row[df_plot_row["test_name"] == param]

                    if df_plot_item.empty:
                        ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', fontsize=12)
                        continue

                    boxes = get_boxes_data(df_plot_item, metric_name, "param_value")

                    if len(boxes) > 0:
                        ax.bxp(boxes, showfliers=False)
                        ax.set_ylim(y_low, y_up)
                        ax.set_xticklabels([b["label"] for b in boxes], rotation=45, ha="right", fontsize=14)

                    if i == 0:
                        ax.set_title(param.replace("_", " ").capitalize())
                    if j == 0:
                        # num_runs = int(df_plot_item["Number of runs"].iloc[0]) if not df_plot_item.empty else 0
                        ax.set_ylabel(f"Metric value ({dataset})")  # \n({num_runs} runs)")

                    ax.set_xlabel("Param value")
                    ax.grid(True)
            else:
                ax = axes[i]

                boxes = get_boxes_data(df_plot_row, metric_name, "test_name")
                # num_runs = int(df_plot_row["Number of runs"].iloc[0]) if not df_plot_row.empty else 0

                if len(boxes) > 0:
                    ax.bxp(boxes, showfliers=False)
                    ax.set_xticklabels([b["label"] for b in boxes], rotation=45, ha="right", fontsize=14)
                    ax.set_title(f"{dataset.capitalize()}", fontsize=14)  # ({num_runs} runs)")
                    ax.set_ylabel(get_metric_title(metric_name), fontsize=14)
                    # ax.set_ylabel("Metric value", fontsize=14)
                    ax.set_ylim(y_low, y_up)
                    ax.grid(True)
                else:
                    ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', fontsize=12)

        metric_name_title = get_metric_title(metric_name)

        fig.suptitle(f"{metric_name_title}\n({method_name})", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95]) # (left, bottom, right, top), 0.95 to have a small margin on top for title

        plot_file_path = os.path.join(out_path, f"{metric_name}.png")
        plt.savefig(plot_file_path)
        plt.close()

def get_bar_width(df, x_key):
    width = 0.6

    if df[x_key].dtype == float or df[x_key].dtype == int:
        width = (df[x_key].max() - df[x_key].min()) / (len(df) * 2)

        # sorted_x = sorted(df[x_key])
        # min_step = min(sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x)-1))
        # width = min_step * 0.8

    return width

def create_subplot(df, x_key, dataset_name, ax, do_set_y_label=True):
    if do_set_y_label:
        # num_runs = f"{int(df['Number of runs'].iloc[0])}" if not df.empty else "?"
        ax.set_ylabel(f"Metric value ({dataset_name})")  # \n({num_runs} runs)")

    if df.empty:
        ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', fontsize=12)
        return ax

    df = df.sort_values(x_key)
    width = get_bar_width(df, x_key)

    ax.bar(
        df[x_key],
        df["success_rate"],
        width=width,
        edgecolor='black'
    )

    ax.set_ylim(-0.05, 1.05)

    # ax.grid(True, axis="y")

    return ax


def get_metric_title(metric_name):
    metric_name_title = re.sub(r'\b(replace|append)\b', '', metric_name.replace('_', ' '), flags=re.IGNORECASE)
    metric_name_title = re.sub(r'\s+', ' ', metric_name_title).strip()
    metric_name_title = re.sub(r'\bnum\b', 'number of', metric_name_title, flags=re.IGNORECASE)
    metric_name_title = re.sub(r'\bsols\b', 'solutions', metric_name_title, flags=re.IGNORECASE)
    metric_name_title = metric_name_title.title()

    return metric_name_title

def gen_plot_aggregate(df, metric_name, out_path, method_name, box_plot=True):
    fig, ax = plt.subplots(figsize=(6.3,6.3))

    y_low, y_up = get_y_lims(df, metric_name)

    if box_plot:
        boxes = get_boxes_data(df, metric_name, "dataset")
        ax.bxp(boxes, showfliers=False)
        ax.set_xticklabels([b["label"] for b in boxes], rotation=45, ha="right", fontsize=14)
        ax.grid(True)
    else:
        if df.empty:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', fontsize=12)
            return ax

        x = []
        for i in range(len(df)):
            x.append(f"{df['dataset'].iloc[i]}")  # ({int(df['Number of runs'].iloc[i])} runs)")

        ax.bar(
            x,
            df[metric_name],
            width=0.8,
            edgecolor='black'
        )

        ax.set_xlabel("Dataset", fontsize=14)
        ax.set_xticks(range(len(df["dataset"])))
        ax.set_xticklabels(x, rotation=45, ha="right", fontsize=14)

    metric_name_title = get_metric_title(metric_name)

    ax.set_ylabel(metric_name_title, fontsize=14)
    # ax.set_ylabel("Metric value", fontsize=14)
    ax.set_ylim(y_low, y_up)


    fig.suptitle(f"{metric_name_title}\n({method_name})", fontsize=16)
    fig.tight_layout(
        rect=[0, 0, 1, 0.95])  # (left, bottom, right, top), 0.95 to have a small margin on top for title

    plot_file_path = os.path.join(out_path, f"{metric_name}.png")
    plt.savefig(plot_file_path)
    plt.close()


def plot_success_rate_lineplots(df, out_path, method_name, aggregate_datasets=False):
    method_name = get_method_title(method_name)

    if df.empty:
        return
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

    if num_cols == 1 and aggregate_datasets and df["test_name"].nunique() == 1:
        metric_name = "success_rate"
        gen_plot_aggregate(df, metric_name, out_path, method_name, box_plot=False)
        return


    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(5*num_cols, 6*num_rows),
        sharey=True,
    )

    if num_rows == 1:
        axes = [axes]
    if col_keys is not None and num_cols == 1:
        axes = [[ax] for ax in axes]

    for i, dataset in enumerate(row_keys):
        df_row = df[df["dataset"] == dataset]

        if col_keys is not None:
            for j, param in enumerate(col_keys):
                df_plot_item = df_row[df_row["test_name"] == param]

                ax = create_subplot(df_plot_item, "param_value", dataset, axes[i][j], j == 0)

                if i == 0:
                    ax.set_title(param.replace("_", " ").capitalize())

                ax.set_xlabel("Param value")

        else:
            ax = create_subplot(df_row, "test_name", dataset, axes[i])

            ax.set_xlabel("Test name")
            ax.set_title(f"Tests on {dataset}")
            ax.set_xticks(range(len(df_row["test_name"])))
            ax.set_xticklabels(df_row["test_name"], rotation=45, ha="right")


    fig.suptitle(f"Success Rates\n({method_name})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # (left, bottom, right, top), 0.95 to have a small margin on top for title

    plot_file_path = os.path.join(out_path, "success_rates.png")
    plt.savefig(plot_file_path)
    plt.close()


def main(filter_params=None, param_values_range=None, exclude_test_names=None, exclude_datasets=None, filter_test_names=None, filter_datasets=None, plot_gradient=True, plot_metaheuristics=True, plot_params=True, plot_general=True, do_plot_metrics=False, do_plot_success_rates=True, aggregate_datasets=False):
    if not (plot_gradient or plot_metaheuristics) or not (plot_params or plot_general):
        print("Nothing to plot. Exiting.")
        return
    data_path_base = "../cf_results"
    out_path_base = "../cf_results/plots"

    method_names = []
    if plot_gradient:
        method_names.append("cf_gradient")
    if plot_metaheuristics:
        method_names.append("cf_metaheuristics")

    config = []
    for method in method_names:
        if plot_general:
            data_path = os.path.join(data_path_base, method)
            out_path = os.path.join(out_path_base, method, "general")
            config.append({
                "data_path": data_path,
                "out_path": out_path,
                "method": method
            })
        if plot_params:
            for num_rules in [1, 2]:
                sub_path = os.path.join(method, "param_search", f"min_num_rules_{num_rules}")
                data_path = os.path.join(data_path_base, sub_path)
                out_path = os.path.join(out_path_base, sub_path)
                config.append({
                    "data_path": data_path,
                    "out_path": out_path,
                    "method": method
                })

    for cfg in config:
        data_path = cfg["data_path"]
        out_path = cfg["out_path"]
        method = cfg["method"]

        if not os.path.exists(data_path):
            print(f"Data path does not exist: {data_path}")
            continue

        if os.path.exists(out_path):
            shutil.rmtree(out_path)

        print(f"Creating plots in {out_path} using data from {data_path}...")

        df = collect_data(data_path, exclude_test_names, filter_test_names, filter_datasets, filter_params, param_values_range, exclude_datasets)

        if do_plot_metrics:
            plot_metrics(df, out_path, method, aggregate_datasets)
        if do_plot_success_rates:
            plot_success_rate_lineplots(df, out_path, method, aggregate_datasets)


if __name__ == "__main__":

    main(filter_test_names=["classic"], exclude_datasets=["magic", "movement_libras"], plot_gradient=True, plot_metaheuristics=True, plot_params=False, plot_general=True, do_plot_metrics=False, do_plot_success_rates=True, aggregate_datasets=True)

    # main(filter_test_names=["classic"], filter_params=["time_in_seconds"], exclude_datasets=["magic", "movement_libras"], plot_gradient=True, plot_metaheuristics=True, plot_params=False, plot_general=True, do_plot_metrics=True, do_plot_success_rates=False, aggregate_datasets=True)

    # main(filter_params=["num_changed_features"], filter_test_names=["classic", "num_features", "num_features_no_change_loss", "less_edits", "no_fs_type_change"], filter_datasets=["bupa", "iris"], plot_gradient=False, plot_metaheuristics=True, plot_params=False, plot_general=True, do_plot_metrics=True, do_plot_success_rates=False, aggregate_datasets=False)
    # main(filter_params=["train_error_rate_append", "train_error_rate_append_initial_variation"], filter_test_names=["classic", "less_edits", "no_fs_type_change", "error_rate"], filter_datasets=["bupa", "iris"], plot_gradient=False, plot_metaheuristics=True, plot_params=False, plot_general=True, do_plot_metrics=True, do_plot_success_rates=False, aggregate_datasets=True)
    # main(filter_params=["train_error_rate_append", "train_error_rate_append_initial_variation"], filter_test_names=["classic", "min_num_rules_2"], filter_datasets=["bupa", "iris"], plot_gradient=False, plot_metaheuristics=True, plot_params=False, plot_general=True, do_plot_metrics=True, do_plot_success_rates=False, aggregate_datasets=True)
    # main(filter_params=["diversity", "num_sols"], exclude_datasets=["magic", "movement_libras"], filter_test_names=["classic"], plot_gradient=False, plot_metaheuristics=True, plot_params=False, plot_general=True, do_plot_metrics=True, do_plot_success_rates=False, aggregate_datasets=True)
    # main(filter_params=["num_changed_features"], exclude_datasets=["magic", "movement_libras"], filter_test_names=["classic"], plot_gradient=True, plot_metaheuristics=False, plot_params=False, plot_general=True, do_plot_metrics=True, do_plot_success_rates=False, aggregate_datasets=True)
    # main(filter_params=["train_error_rate_append", "train_error_rate_append_initial_variation"], exclude_datasets=["magic", "movement_libras"], filter_test_names=["classic"], plot_gradient=True, plot_metaheuristics=False, plot_params=False, plot_general=True, do_plot_metrics=True, do_plot_success_rates=False, aggregate_datasets=True)

    # main(filter_test_names=["classic"], filter_datasets=["iris", "magic", "movement_libras"], plot_gradient=True, plot_metaheuristics=True, plot_params=False, plot_general=True, do_plot_metrics=True, do_plot_success_rates=True, aggregate_datasets=True)

    # main(filter_datasets=["iris", "bupa", "pima"], filter_params=["mutation_prob"], param_values_range=(0.1, 1), plot_gradient=False, plot_metaheuristics=True, plot_params=True, plot_general=False, do_plot_metrics=True, do_plot_success_rates=True, aggregate_datasets=False)
    # main(filter_datasets=["iris", "bupa", "pima"], filter_params=["mutation_prob"], param_values_range=(0.01, 0.1), plot_gradient=False, plot_metaheuristics=True, plot_params=True, plot_general=False, do_plot_metrics=True, do_plot_success_rates=True, aggregate_datasets=False)
    # main(filter_datasets=["iris", "bupa", "pima"], filter_params=["mutation_prob"], param_values_range=(0, 0.01), plot_gradient=False, plot_metaheuristics=True, plot_params=True, plot_general=False, do_plot_metrics=True, do_plot_success_rates=True, aggregate_datasets=False)
    # main(filter_datasets=["iris", "bupa", "pima"], plot_gradient=False, plot_metaheuristics=True, plot_params=True, plot_general=False, do_plot_metrics=False, do_plot_success_rates=True, aggregate_datasets=False)

    # main(filter_params=["train_error_rate_append_initial_variation"], filter_test_names=["classic"], plot_gradient=False, plot_metaheuristics=True, plot_params=False, plot_general=True, do_plot_metrics=True, do_plot_success_rates=False, aggregate_datasets=True)

    # main( filter_test_names=["X_crowding", "classic"], plot_gradient=False, plot_metaheuristics=True, plot_params=True, plot_general=True, do_plot_metrics=True, do_plot_success_rates=True, aggregate_datasets=False)
    # main(filter_test_names=["classic"], plot_gradient=True, plot_metaheuristics=True, plot_params=True, plot_general=True, do_plot_metrics=True, do_plot_success_rates=True, aggregate_datasets=True)
