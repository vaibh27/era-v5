import torch
import torch.distributed as dist


def shard_size(numel, world_size):
    """Size of one equal shard once numel is padded up to a multiple of world_size."""
    return (numel + world_size - 1) // world_size


def flatten_pad(tensor, world_size):
    flat = tensor.reshape(-1)
    pad = shard_size(flat.numel(), world_size) * world_size - flat.numel()
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat


def local_shard(tensor, rank, world_size):
    """This rank's 1/world_size slice of `tensor`, taken from its padded flat view."""
    flat = flatten_pad(tensor, world_size)
    s = shard_size(tensor.numel(), world_size)
    return flat[rank * s:(rank + 1) * s].clone()


def all_gather_flat(local_chunk, world_size, pg=None):
    """Gather equal-sized local chunks from all ranks -> one concatenated flat tensor."""
    out = [torch.empty_like(local_chunk) for _ in range(world_size)]
    dist.all_gather(out, local_chunk.contiguous(), group=pg)
    return torch.cat(out)


def reduce_scatter_flat(full_padded_flat, world_size, pg=None, average=True):
    """Scatter-reduce a full padded flat tensor -> this rank's reduced local shard."""
    s = full_padded_flat.numel() // world_size
    chunks = [full_padded_flat[i * s:(i + 1) * s].contiguous() for i in range(world_size)]
    out = torch.empty_like(chunks[0])
    dist.reduce_scatter(out, chunks, group=pg)
    if average:
        out /= world_size
    return out


def unpad_reshape(flat, shape):
    n = 1
    for d in shape:
        n *= d
    return flat[:n].reshape(shape)
