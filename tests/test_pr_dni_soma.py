import unittest

import torch


from models.Modules.pr_dni_soma import PRDNISoma, clip_round_ste


class PRDNISomaTest(unittest.TestCase):
    def test_forward_returns_normalized_integer_activation_and_state_shapes(self):
        soma = PRDNISoma(channels=4, capacity=8, pre_norm="identity", gate_hidden_channels=4)
        x = torch.rand(3, 2, 4, 5, 6)

        state = soma(x, return_state=True)

        self.assertEqual(state["S"].shape, x.shape)
        self.assertEqual(state["K"].shape, x.shape)
        self.assertEqual(state["D"].shape, (3, 2, 1, 1, 1))
        self.assertEqual(state["V_pre"].shape, x.shape)
        self.assertEqual(state["U"].shape, x.shape)
        self.assertEqual(state["V_post"].shape, x.shape)
        self.assertEqual(state["c"].shape, (3, 2, 1, 5, 6))
        self.assertEqual(state["rho"].shape, (3, 2, 1, 5, 6))
        self.assertEqual(state["gamma"].shape, (3, 2, 1, 5, 6))
        self.assertEqual(state["relation_cue"].shape, (3, 2, 5, 5, 6))
        self.assertTrue(torch.all(state["S"] >= 0))
        self.assertTrue(torch.all(state["S"] <= 1))
        self.assertTrue(torch.equal(state["K"], state["K"].round()))
        self.assertTrue(torch.all(state["D"] == 8))

    def test_clip_round_ste_passes_gradients_inside_clamp_range(self):
        x = torch.tensor([0.2, 1.7, 4.3], requires_grad=True)

        y = clip_round_ste(x, 0.0, 8.0)
        y.sum().backward()

        self.assertTrue(torch.equal(y.detach(), torch.tensor([0.0, 2.0, 4.0])))
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))

    def test_first_phase_has_no_carry_and_module_has_no_persistent_membrane_buffers(self):
        soma = PRDNISoma(channels=2, capacity=8, pre_norm="identity")
        x = torch.rand(1, 1, 2, 3, 3)

        state = soma(x, return_state=True)

        self.assertTrue(torch.allclose(state["U"][0], state["V_pre"][0]))
        self.assertEqual(list(soma.buffers()), [])

    def test_gate_maps_high_contamination_to_lower_retention_and_stronger_reset(self):
        soma = PRDNISoma(
            channels=2,
            capacity=8,
            pre_norm="identity",
            gate_hidden_channels=0,
            rho_min=0.1,
            rho_max=0.9,
            gamma_min=0.2,
            gamma_max=1.0,
        )
        with torch.no_grad():
            soma.gate[0].weight.fill_(1.0)
            soma.gate[0].bias.zero_()

        low_cue = torch.zeros(1, 5, 2, 2)
        high_cue = torch.ones(1, 5, 2, 2)

        low = soma.gate_from_relation_cue(low_cue)
        high = soma.gate_from_relation_cue(high_cue)

        self.assertTrue(torch.all(high["c"] > low["c"]))
        self.assertTrue(torch.all(high["rho"] < low["rho"]))
        self.assertTrue(torch.all(high["gamma"] > low["gamma"]))

    def test_forward_is_adjacent_causal_and_does_not_read_future_phase(self):
        soma = PRDNISoma(channels=3, capacity=8, pre_norm="identity", gate_hidden_channels=0)
        x = torch.rand(3, 1, 3, 4, 4)
        x_changed_future = x.clone()
        x_changed_future[2] = x_changed_future[2] + 10.0

        first = soma(x, return_state=True)
        second = soma(x_changed_future, return_state=True)

        self.assertTrue(torch.allclose(first["S"][:2], second["S"][:2]))
        self.assertTrue(torch.allclose(first["K"][:2], second["K"][:2]))
        self.assertTrue(torch.allclose(first["V_post"][:2], second["V_post"][:2]))

    def test_mixed_skip_feature_concatenates_integer_activation_and_membrane_proxy(self):
        soma = PRDNISoma(channels=3, capacity=8, pre_norm="identity")
        x = torch.rand(2, 1, 3, 4, 4)

        state = soma(x, return_state=True, return_mixed=True)

        self.assertEqual(state["mixed"].shape, (2, 1, 6, 4, 4))
        self.assertTrue(torch.allclose(state["mixed"][:, :, :3], state["S"]))


if __name__ == "__main__":
    unittest.main()
