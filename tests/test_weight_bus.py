"""The shared-memory weight bus must deliver exactly what the queue fan-out did.

The point of :class:`WeightBus` is that broadcast cost stops scaling with actor
count; the point of these tests is that it still moves the *same bytes*. A weight
transport that silently drops or garbles an entry would show up as a mysterious
training regression, not as a crash.
"""

from __future__ import annotations

import torch
import torch.multiprocessing as mp

from deepnash_rbc.config import Config
from deepnash_rbc.network import make_net
from deepnash_rbc.weights import WeightBus


def _small_cfg(arch: str = "resnet") -> Config:
    cfg = Config()
    cfg.encoding.history = 2
    cfg.network.arch = arch
    cfg.network.channels = 8
    cfg.network.blocks = 1
    cfg.network.enc_blocks = 1
    cfg.network.mixer_dim = 16
    cfg.network.mixer_layers = 1
    cfg.network.nhead = 2
    return cfg


def _assert_same_state(a: torch.nn.Module, b: torch.nn.Module) -> None:
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"{key} differs"


def test_publish_load_roundtrip_is_exact():
    cfg = _small_cfg()
    torch.manual_seed(0)
    source = make_net(cfg.encoding, cfg.network)
    torch.manual_seed(1)
    target = make_net(cfg.encoding, cfg.network)

    bus = WeightBus.create(source, mp.get_context("spawn"))
    bus.publish(source)
    assert bus.load_into(target) is True
    _assert_same_state(source, target)


def test_load_is_a_noop_until_the_next_publish():
    """Actors poll this every batch; it must be free when nothing has changed."""
    cfg = _small_cfg()
    net = make_net(cfg.encoding, cfg.network)
    bus = WeightBus.create(net, mp.get_context("spawn"))
    bus.publish(net)

    assert bus.load_into(net) is True    # first read of generation 1
    assert bus.load_into(net) is False   # nothing new
    bus.publish(net)
    assert bus.load_into(net) is True    # generation 2


def test_second_publish_overwrites_the_slab():
    cfg = _small_cfg()
    torch.manual_seed(0)
    source = make_net(cfg.encoding, cfg.network)
    target = make_net(cfg.encoding, cfg.network)
    bus = WeightBus.create(source, mp.get_context("spawn"))

    bus.publish(source)
    bus.load_into(target)
    with torch.no_grad():
        for p in source.parameters():
            p.add_(1.0)
    bus.publish(source)
    bus.load_into(target)
    _assert_same_state(source, target)


def test_layout_covers_non_float_buffers():
    """BatchNorm's num_batches_tracked is int64 -- the slab is byte-addressed
    precisely so mixed dtypes survive the round trip."""
    cfg = _small_cfg()
    net = make_net(cfg.encoding, cfg.network)
    counters = [k for k in net.state_dict() if k.endswith("num_batches_tracked")]
    assert counters, "expected int64 entries in the resnet state dict"

    net.train()
    net(torch.zeros(2, cfg.encoding.in_channels, 8, 8))  # bumps the counters
    net.eval()

    target = make_net(cfg.encoding, cfg.network)
    bus = WeightBus.create(net, mp.get_context("spawn"))
    bus.publish(net)
    bus.load_into(target)
    for key in counters:
        assert target.state_dict()[key].dtype == torch.int64
    _assert_same_state(net, target)


def _child(bus, cfg, out_q):
    net = make_net(cfg.encoding, cfg.network)
    bus.wait_for_weights(net, timeout=60.0)
    # numpy, not tensors: torch.multiprocessing sends tensors by passing a shared
    # -memory fd, which needs this process to outlive the parent's receive.
    out_q.put({k: v.numpy().copy() for k, v in net.state_dict().items()})


def test_bus_survives_the_spawn_pickle():
    """The numpy views don't survive spawn; the slab, lock and counter must."""
    ctx = mp.get_context("spawn")
    cfg = _small_cfg()
    torch.manual_seed(3)
    net = make_net(cfg.encoding, cfg.network)
    bus = WeightBus.create(net, ctx)
    bus.publish(net)

    out_q = ctx.Queue()
    proc = ctx.Process(target=_child, args=(bus, cfg, out_q), daemon=True)
    proc.start()
    try:
        received = out_q.get(timeout=120)
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()

    sd = net.state_dict()
    assert received.keys() == sd.keys()
    for key in sd:
        assert torch.equal(sd[key], torch.from_numpy(received[key])), \
            f"{key} differs across the spawn"
