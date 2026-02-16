import unittest

from yoctoGPT.model import GPT, GPTConfig
from yoctoGPT.optim import build_weight_decay_param_groups


class TestOptimParamGroups(unittest.TestCase):
    def test_weight_decay_grouping(self) -> None:
        model = GPT(GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=16))
        groups = build_weight_decay_param_groups(model, weight_decay=0.1)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["weight_decay"], 0.1)
        self.assertEqual(groups[1]["weight_decay"], 0.0)

        decay_params = set(map(id, groups[0]["params"]))
        no_decay_params = set(map(id, groups[1]["params"]))

        # No parameter should appear in both groups.
        self.assertEqual(len(decay_params.intersection(no_decay_params)), 0)

        # Every trainable parameter should appear once.
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(trainable, decay_params.union(no_decay_params))


if __name__ == "__main__":
    unittest.main()
