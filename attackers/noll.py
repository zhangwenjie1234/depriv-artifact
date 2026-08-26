from attackers.vflbase import BaseVFL

class Attacker(BaseVFL):
    def __init__(self, args, entire_model, train_dataset, test_dataset):
        super().__init__(args, entire_model, train_dataset, test_dataset)
        raise NotImplementedError(
            "attack='noll' is listed in the CLI, but it is not implemented in the current branch. "
            "There is no attack execution or attack-metric logging path for 'noll' right now."
        )
