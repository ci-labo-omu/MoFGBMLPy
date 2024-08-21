import os

from mofgbmlpy.utility.parallel_cross_validation import show_results_box_plot, get_results, run_cross_validation, \
    task_nsgaii_homo_triangle_2_3_4_5


def test_runs_iris():
    data_name = "iris"
    root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_root_folder = root_folder + os.sep + "results"
    algorithm_id = f"Basic{data_name}Basic"
    dataset_root = root_folder + os.sep + "dataset"

    args = [
        "--data-name", data_name,
        "--rand-seed", "2020",
        "--terminate-evaluation", "300",
        "--objectives", "num-rules", "error-rate",
        "--algorithm-id", algorithm_id,
        "--root-folder", results_root_folder
    ]

    run_cross_validation(args, dataset_root, task_nsgaii_homo_triangle_2_3_4_5)
    results = get_results(results_root_folder, algorithm_id, data_name)

    results_solutions = [run["solutions"] for run in results]
    _ = [run["exec_time"] for run in results]

    show_results_box_plot(results_solutions, "num_rules", title="Cross validation on Iris with NSGA-II")



