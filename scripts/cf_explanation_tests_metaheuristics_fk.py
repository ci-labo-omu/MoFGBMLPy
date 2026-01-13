import copy
import os

import numpy as np

from mofgbmlpy.data.dataset import Dataset

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from pymoo.core.population import Population

from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.data.pattern import Pattern

from mofgbmlpy.data.output import Output
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain
from mofgbmlpy.explainer.util import remove_duplicates
from mofgbmlpy.explainer.counterfactual_explainer_benchmark import CounterFactualExplainerBenchmark as CFEBenchmark
from mofgbmlpy.explainer.counterfactual_explainer_metaheuristics_fk import CounterFactualExplainerMetaheuristicsFK as CFEMetaheuristicsFK
from mofgbmlpy.explainer.util import append_rule_classifier
from scripts.cf_explanation_tests_metaheuristics import get_config

if __name__ == "__main__":
    # Example on Pima
    _, train_dataset, test_dataset, non_dominated_solutions, _ = get_config("pima", min_num_rules=2, verbose=True,
                                                                            interpretability_obj="total-rule-length")
    cl_idx = 6
    initial_classifier = non_dominated_solutions[cl_idx][0]
    decision_boundaries_fixed_vals = [0.18, None, 0.59, 0.23, 0.03, None, 0.13, 0.13]

    for var in initial_classifier.get_vars():
        print(var.get_rule())

    var_names = ["Preg", "Plas", "Pres", "Skin", "Insu", "Mass", "Pedi", "Age"]
    class_labels = ["Tested negative", "Tested positive"]
    cf_rule_1 = CFEBenchmark.main_plot_single(CFEMetaheuristicsFK, non_dominated_solutions, plot=False, var_names=var_names,
                                  class_labels=class_labels, test_dataset=test_dataset, cl_idx=cl_idx, r_idx=0,
                                  c_target=1, sol_idx=0, sampling_change_prob=0.2, mutation_prob=0.7, crossover_prob=0.7,
                                  objectives=["confidence_loss", "change_loss"],
                                  decision_boundaries_fixed_vals=decision_boundaries_fixed_vals)[0]

    print("Counterfactual Rule 1:", cf_rule_1)

    new_cl = append_rule_classifier(initial_classifier, cf_rule_1, train_set=train_dataset)
    non_dominated_solutions = np.array([[new_cl]])

    cf_rule_2 = CFEBenchmark.main_plot_single(CFEMetaheuristicsFK, non_dominated_solutions, plot=False, var_names=var_names,
                                  class_labels=class_labels, test_dataset=test_dataset, cl_idx=0, r_idx=1, c_target=0,
                                  sol_idx=0, sampling_change_prob=0.2, mutation_prob=0.7, crossover_prob=0.7,
                                  objectives=["confidence_loss", "change_loss"],
                                  decision_boundaries_fixed_vals=decision_boundaries_fixed_vals)[0]
    new_cl = append_rule_classifier(new_cl, cf_rule_2, train_set=train_dataset)

    rules_names = ["Factual Rule 1", "Factual Rule 2", "CF Rule 1", "CF Rule 2"]

    new_cl.set_vars(new_cl.get_vars()[2:])

    new_cl.update_winners_and_errors(train_dataset)
    new_cl.plot_rules(var_names, rules_names, dims=[1, 3, 5])

    print(f"Initial classifier error rate: {initial_classifier.get_error_rate()}")
    print(f"New classifier error rate: {new_cl.get_error_rate()}")
    for var in new_cl.get_vars():
        print(var.get_rule())

    print("Initial classifier:", initial_classifier)
    print("New classifier:", new_cl)

    ######################"
    #
    # test_configs = {
    #     "classic": {},
    # }
    # test_names = list(test_configs.keys())
    #
    # # for data_name in ["appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar", "magic", "movement_libras"]:
    # for data_name in ["iris"]:
    #     all_tests_already_exist = True
    #     for test_name in test_names:
    #         test_path = f"..\\cf_results_fk\\cf_metaheuristics\\{test_name}\\{data_name}"
    #         if not os.path.exists(test_path):
    #             all_tests_already_exist = False
    #             break
    #     if all_tests_already_exist:
    #         print(f"All tests on {data_name} have already been run, skipping...")
    #         continue
    #
    #     num_classes, _, test_dataset, non_dominated_solutions, _ = get_config(data_name)
    #
    #     for test_name, config in test_configs.items():
    #         result_path = f"..\\cf_results_fk\\cf_metaheuristics\\{test_name}\\{data_name}"
    #         CFEBenchmark.main_benchmark(CFEMetaheuristics, num_classes, non_dominated_solutions, out_path=result_path, test_dataset=test_dataset, data_name=data_name, test_name=test_name, **config)
