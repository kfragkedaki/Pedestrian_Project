from SINDDataset import SinD

class PedestrianData(object):
    NAME = "sind"

    @staticmethod
    def make_dataset(*args, **kwargs):
        sind = SinD()
        return sind.split_pedestrian_data(chunk_size=args['chunk_size'], padding_value=args['padding_value'])

    # @staticmethod
    # def make_state(*args, **kwargs):
    #     return StateEVRP.initialize(*args, **kwargs)
