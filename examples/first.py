from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_5 import HomoTriangleKnowledgeFactory_5
from mofgbmlpy.main.moead.mofgbml_moead_main import MoFGBMLMOEADMain
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_main import MoFGBMLNSGAIIMain

# runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
# results = runner.main(["--help"])
args = [
    "--algorithm-id", "1",
    "--experiment-id", "2",
    "--data-name", "pima",
    "--train-file", "../dataset/pima/a0_0_pima-10tra.dat",
    "--test-file", "../dataset/pima/a0_0_pima-10tst.dat",
    "--terminate-evaluation", "10000",
    "--objectives", "total-rule-length", "error-rate",
    # "--crossover-type", "pittsburgh-crossover",
    # "--antecedent-factory", "all-combination-antecedent-factory",
    "--no-output-files",
    "--verbose",
]
runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_5)
results = runner.main(args)

min_length = results.opt.get("X")[0, 0].get_var(0).get_rule().get_length()
max_length = min_length
for sol in results.opt.get("X")[:, 0]:
    for var in sol.get_vars():
        length = var.get_rule().get_length()
        if length < min_length:
            min_length = length
        elif length > max_length:
            max_length = length
print(min_length, max_length)

i = 1
for var in results.opt.get("X")[0, 0].get_vars():
    print(f"{i}:\t{var.get_rule().get_linguistic_representation()}")
    i += 1


plot = runner.get_pareto_front_plot(results.opt)
plot.show()
# plot.ax.set_ylim([0,1])
plot.ax.grid(visible=True)

results.opt.get('X')[1,0]

runner.plot_line_interpretability_error_rate_tradeoff(results.opt.get('X')[:, 0], title="MoFGBMLPy on Pima with NSGA-II", xlim=[0,51])
# runner.plot_line_interpretability_error_rate_tradeoff(results.opt.get('X')[:, 0], title="MoFGBMLPy on Iris with NSGA-II", xlim=[0,16])
# runner.plot_line_interpretability_error_rate_tradeoff(results.opt.get('X')[:, 0], title="MoFGBMLPy on Iris with MOEA/D", xlim=[0,10])
runner.plot_fuzzy_variables()