import copy
from types import SimpleNamespace

import pytest
import torch

from prompt_a import adapt_batch, mask_graph, load_backbone


class Graph:
    def __init__(self, label=0):
        self.x = torch.tensor([[6], [7], [8], [6], [1]])
        self.ptr = torch.tensor([0, 3, 4, 5])
        self.batch = torch.tensor([0, 0, 0, 1, 2])
        self.y = torch.tensor([label, label, label])
        self.domain_id = torch.tensor([label] * 3)

    def clone(self):
        return copy.deepcopy(self)


class TinyFrozen(torch.nn.Module):
    """Differentiable substitute to test the adaptation protocol without PyG."""
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(40, 4)
        self.aux = torch.nn.Linear(4, 40)
        self.head = torch.nn.Linear(4, 2)
        self.register_buffer('sentinel', torch.ones(2))

    def forward(self, graph, return_node_feats=False, graph_prompt=None):
        assert not hasattr(graph, 'y') and not hasattr(graph, 'domain_id')
        nodes = self.embed(graph.x[:, 0]) + graph_prompt
        if return_node_feats:
            return self.aux(nodes)
        return self.head(torch.stack([nodes[graph.batch == i].mean(0) for i in range(3)]))


def test_mask_retains_visible_atoms_and_does_not_modify_input():
    graph = Graph()
    masked, idx, labels = mask_graph(graph, .2, torch.Generator().manual_seed(1))
    assert idx.numel() == 1 and idx.item() < 3
    assert torch.equal(labels, graph.x[idx, 0])
    assert not (graph.x == 39).any()
    assert (masked.x == 39).sum() == 1


def test_labels_do_not_affect_update_and_only_prompt_changes():
    torch.manual_seed(5)
    model = TinyFrozen()
    snapshot = copy.deepcopy(model.state_dict())
    args = SimpleNamespace(steps=2, mask_rate=.2, cons_weight=1.,
                           anchor_weight=.001, max_step_norm=.005)
    outputs, prompts = [], []
    for label in (0, 1):
        p = torch.nn.Parameter(torch.zeros(4))
        optimizer = torch.optim.Adam([p], lr=.1)
        probs, _ = adapt_batch(model, Graph(label), p, optimizer,
                              torch.Generator().manual_seed(10), args, torch.zeros(4))
        outputs.append(probs)
        prompts.append(p.detach().clone())
        assert 0 < p.detach().norm() <= args.max_step_norm + 1e-6
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in snapshot.items())
        assert all(v.grad is None for v in model.parameters())
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(prompts[0], prompts[1])


def test_real_framework_zero_prompt_and_gradient():
    pytest.importorskip('torch_geometric')
    pytest.importorskip('torch_scatter')
    from torch_geometric.data import Data
    from models.Framework import Framework
    from prompt_a import collate_graphs, freeze
    graphs = []
    for n in (3, 4):
        edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        sub = Data(x=torch.full((n, 1), 6), edge_index=edge,
                   edge_attr=torch.ones((4, 1), dtype=torch.long))
        graph = sub.clone()
        graph.subs = [sub.clone(), sub.clone()]
        graphs.append(graph)
    data = collate_graphs(graphs)
    model = Framework(128, 128, 2)
    freeze(model)
    snapshot = copy.deepcopy(model.state_dict())
    p = torch.nn.Parameter(torch.zeros(128))
    assert torch.equal(model(data), model(data, graph_prompt=p))
    model(data, graph_prompt=p).square().sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
    assert all(v.grad is None for v in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in snapshot.items())


def test_checkpoint_expands_only_mask_column_and_rejects_other_mismatches(tmp_path):
    class Weights:
        def __init__(self):
            self.values = {'base_model.node_embeddings.0.linears.0.weight': torch.randn(4, 40),
                           'predictor.weight': torch.randn(2, 4), 'aux_predictor.weight': torch.randn(40, 4)}

        def state_dict(self):
            return self.values.copy()

        def load_state_dict(self, state, strict=True):
            self.values = state

        def eval(self):
            return self

        def requires_grad_(self, flag):
            return self

    model = Weights()
    name = 'base_model.node_embeddings.0.linears.0.weight'
    source = {name: torch.randn(4, 39), 'predictor.weight': torch.randn(2, 4)}
    path = tmp_path / 'checkpoint.pt'
    torch.save({'main': source}, path)
    report = load_backbone(model, path)
    assert report['expanded'] == [name]
    assert torch.equal(model.values[name][:, :39], source[name])
    assert torch.count_nonzero(model.values[name][:, 39]) == 0
    source['predictor.weight'] = torch.randn(3, 4)
    torch.save({'main': source}, path)
    with pytest.raises(ValueError, match='Incompatible weight'):
        load_backbone(model, path)
