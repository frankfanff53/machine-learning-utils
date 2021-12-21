class Classifier(object):
    """
    Abstract classifier class.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def predict(self, *args, **kwargs) -> float:
        raise NotImplementedError()
