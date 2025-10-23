from typing import Tuple, List, Dict
import numpy as np
import torch
from preprocess import iqa
from utils import ncc_loss, Stack, DeviceType


def matrix_rank(
    stacks: List[Stack], threshold: float = 0.1, rank_only: bool = False
) -> List[float]:
    scores: List[float] = []
    for stack in stacks:
        masked_slices = stack.slices * stack.mask.float()
        score = _matrix_rank(masked_slices.flatten(1, -1), threshold, rank_only)
        scores.append(score)
    return scores


def _matrix_rank(mat: torch.Tensor, threshold: float, rank_only: bool) -> float:
    s = torch.linalg.svdvals(mat)
    s = s / s[0]
    R = torch.count_nonzero(s > 1e-6)
    norm2 = torch.cumsum(s.pow(2), 0)
    norm2_all = norm2[-1]
    e = threshold
    for r in range(len(s) - 1):
        if norm2[r] > (1 - threshold * threshold) * norm2_all:
            e = torch.sqrt((norm2_all - norm2[r]) / norm2_all).item()
            break
    return float((r + 1) / R if rank_only else e * (r + 1) / R / threshold)


def ncc(stacks: List[Stack]) -> List[float]:
    scores: List[float] = []
    for stack in stacks:
        masked_slices = stack.slices * stack.mask.float()
        score = _ncc(masked_slices)
        scores.append(score)
    return scores


def _ncc(slices: torch.Tensor) -> float:
    slices1 = slices[:-1]
    slices2 = slices[1:]
    mask = ((slices1 + slices2) > 0).float()
    ncc = ncc_loss(slices1, slices2, mask, win=None, reduction="none")
    ncc_weight = mask.sum((1, 2, 3))
    ncc_weight = ncc_weight.view(ncc.shape)
    return float(-((ncc * ncc_weight).sum() / ncc_weight.sum()).item())


def compute_metric(
    stacks: List[Stack],
    metric: str,
    batch_size: int,
    augmentation: bool,
    device: DeviceType,
) -> Tuple[List[float], bool]:
    descending = True
    if metric == "ncc":
        scores = ncc(stacks)
    elif metric == "matrix-rank":
        scores = matrix_rank(stacks)
        descending = False
    elif metric == "volume":
        scores = [
            int(
                stack.mask.float().sum().item()
                * stack.resolution_x
                * stack.resolution_y
                * stack.gap
            )
            for stack in stacks
        ]
    elif metric == "iqa2d":
        scores = iqa.iqa2d(
            stacks, device, batch_size=batch_size, augmentation=augmentation
        )
    elif metric == "iqa3d":
        scores = iqa.iqa3d(stacks, batch_size=batch_size, augmentation=augmentation)
    else:
        raise ValueError("unkown metric for stack assessment")

    return scores, descending


def sort_and_filter(
    stacks: List[Stack],
    scores: List[float],
    descending: bool,
    filter_method: str,
    cutoff: float,
) -> Tuple[List[Stack], List[int], List[bool]]:
    n_total = len(scores)
    n_keep = n_total
    if filter_method == "top":
        n_keep = min(n_keep, int(cutoff))
    elif filter_method == "bottom":
        n_keep = max(0, n_total - int(cutoff))
    elif filter_method == "percentage":
        n_keep = n_total - int(n_total * min(max(0, cutoff), 1))
    elif filter_method == "threshold":
        if descending:
            n_keep = sum(score >= cutoff for score in scores)
        else:
            n_keep = sum(score <= cutoff for score in scores)
    elif filter_method == "none":
        pass
    else:
        raise ValueError("unknown filter method")

    sorter = np.argsort(-np.array(scores) if descending else scores)
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    ranks = [int(rank) for rank in inv]
    excluded = [rank >= n_keep for rank in ranks]

    output_stacks = [stacks[i] for i in sorter[:n_keep]]

    return output_stacks, ranks, excluded


def assess(
    stacks: List[Stack],
    metric: str,
    filter_method: str,
    cutoff: float,
    batch_size: int,
    augmentation: bool,
    device: DeviceType,
) -> Tuple[List[Stack], List[Dict]]:
    if metric == "none":
        return stacks, []

    scores, descending = compute_metric(
        stacks, metric, batch_size, augmentation, device
    )
    filtered_stacks, ranks, excludeds = sort_and_filter(
        stacks, scores, descending, filter_method, cutoff
    )

    results = []
    for i, (score, rank, excluded, stack) in enumerate(
        zip(scores, ranks, excludeds, stacks)
    ):
        results.append(
            dict(
                input_id=i,
                name=stack.name,
                score=score,
                rank=rank,
                excluded=excluded,
                descending=descending,
            )
        )

    return filtered_stacks, results
