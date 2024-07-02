from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.knowledge.homo_triangle_knowledge_factory import HomoTriangleKnowledgeFactory
from mofgbmlpy.fuzzy.knowledge.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5


def test_create2_3_4_5_plot():
    training_data_file_name = "../dataset/iris/a0_0_iris-10tra.dat"
    training_data = Input.input_data_set(training_data_file_name, False)
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_data.get_num_dim()).create()

    knowledge.plot_all_dim()
    knowledge.plot_one_fuzzy_set(0,1)
    knowledge.plot_one_dim_unique_graph(0)
    knowledge.plot_one_dim_separate_graphs(0)

    assert True