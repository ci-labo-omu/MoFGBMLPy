from mofgbmlpy.main.basic.mofgbml_basic_main import MoFGBMLBasicMain


def test_main():
    args = [
        "iris",
        1,
        2,
        1,
        "../dataset/iris/a0_0_iris-10tra.dat",
        "../dataset/iris/a0_0_iris-10tst.dat",
    ]

    MoFGBMLBasicMain.main(args)
    assert True
