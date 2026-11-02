import unittest
import torch
from scripts.ncm.packed_training import layout, mask_and_positions, convolution
from scripts.ncm.shared_training import layout as single_layout


def additive_recurrence(q, k, v, g, beta, initial_state=None, output_final_state=False, cu_seqlens=None, **kwargs):
    parts, states = [], []
    for i, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        part = q[:, int(start):int(end)].cumsum(1)
        if initial_state is not None:
            part = part + initial_state[i][None, None]
        parts.append(part)
        states.append(part[0, -1])
    return torch.cat(parts, 1), torch.stack(states) if output_final_state else None


class PackedTests(unittest.TestCase):
    def test_recurrent_values_and_gradients_match_independent_examples(self):
        prefixes, sizes = [64, 69, 131], [4, 16, 64, 128]
        lengths = [prefix + sum(sizes) for prefix in prefixes]
        x = torch.randn(1, sum(lengths), 1, 2, dtype=torch.float64, requires_grad=True)
        packed = layout(prefixes, sizes, 'cpu').gated_delta_rule(additive_recurrence, x, x, x, x, x)
        parts, start = [], 0
        for prefix, length in zip(prefixes, lengths):
            piece = x[:, start:start + length]
            parts.append(single_layout(prefix, sizes, 'cpu').gated_delta_rule(
                additive_recurrence, piece, piece, piece, piece, piece))
            start += length
        reference = torch.cat(parts, 1)
        torch.testing.assert_close(packed, reference)
        probe = torch.randn_like(packed)
        a = torch.autograd.grad((packed * probe).sum(), x, retain_graph=True)[0]
        b = torch.autograd.grad((reference * probe).sum(), x)[0]
        torch.testing.assert_close(a, b)

    def test_attention_and_rotary_positions_do_not_cross_conversations(self):
        prefixes, sizes = [64, 69], [4, 16]
        mask, positions = mask_and_positions(prefixes, sizes, 'cpu', torch.float32)
        boundary = prefixes[0] + sum(sizes)
        self.assertTrue(torch.isneginf(mask[:, :, :boundary, boundary:]).all())
        self.assertTrue(torch.isneginf(mask[:, :, boundary:, :boundary]).all())
        self.assertEqual(positions[0, 0, boundary].item(), 0)
        self.assertEqual(positions[0, 0, boundary + 69].item(), 69)
        self.assertEqual(positions[0, 0, boundary + 69 + 4].item(), 69)

    def test_convolution_has_no_cross_example_or_level_history(self):
        prefixes, sizes = [64, 69], [4, 16]
        x = torch.randn(1, sum(prefixes) + 2 * sum(sizes), 1, dtype=torch.float64).transpose(1, 2)
        weight = torch.randn(1, 4, dtype=torch.float64)

        def kernel(value, weight, bias, activation=None):
            return torch.nn.functional.conv1d(value, weight[:, None], padding=3)[:, :, :value.shape[-1]]

        result = convolution(kernel, x, weight, None, None, prefixes, sizes)
        boundary = prefixes[0] + sum(sizes)
        changed = x.clone()
        changed[:, :, :boundary] *= 100
        rerun = convolution(kernel, changed, weight, None, None, prefixes, sizes)
        torch.testing.assert_close(result[:, :, boundary:], rerun[:, :, boundary:])
        changed = x.clone()
        changed[:, :, 64:68] *= 100
        rerun = convolution(kernel, changed, weight, None, None, prefixes, sizes)
        torch.testing.assert_close(result[:, :, 68:boundary], rerun[:, :, 68:boundary])

    def test_short_prefix_cannot_enter_unqualified_packing(self):
        with self.assertRaises(ValueError):
            layout([64, 63], [4, 16], 'cpu')


if __name__ == '__main__':
    unittest.main()
