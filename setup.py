from distutils.core import setup

setup(
    name="mldecisiontreecoursework",
    version="1.0.0",
    packages=[
        "data_loading",
        "decision_tree",
        "entropy",
        "evaluation",
        "visualisation",
    ],
    install_requires=["matplotlib>=3.3.2", "numpy>=1.19.2"],
)
