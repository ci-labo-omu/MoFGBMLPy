from mofgbmlpy.main.moead.mofgbml_moead_main import MoFGBMLMOEADMain


def test_main():
    args = [
        "iris",
        1,
        2,
        1,
        "../dataset/iris/a0_0_iris-10tra.dat",
        "../dataset/iris/a0_0_iris-10tst.dat",
    ]

    MoFGBMLMOEADMain.main(args)
    assert True
