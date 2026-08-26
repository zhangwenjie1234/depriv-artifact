import json
from pathlib import Path

import torch


def load_public_cluster_scores(path, dataset, num_parties):
    """Load per-party public Cluster scores from raw or summary probe JSON."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("Public-risk JSON was not found: {}".format(source))
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Public-risk JSON must contain a list of result rows.")

    grouped = [[] for _ in range(int(num_parties))]
    value_keys = (
        "public_risk_mean_pct",
        "public_risk_pct",
        "cluster_accuracy_mean_pct",
        "cluster_accuracy_pct",
    )
    for row in rows:
        if not isinstance(row, dict) or row.get("dataset") != dataset:
            continue
        party_id = int(row.get("party_id", -1))
        if party_id < 0 or party_id >= int(num_parties):
            continue
        value = None
        for key in value_keys:
            if row.get(key) is not None:
                value = float(row[key])
                break
        if value is not None:
            grouped[party_id].append(value)

    missing = [party_id for party_id, values in enumerate(grouped) if not values]
    if missing:
        raise ValueError(
            "Public-risk JSON has no {} scores for party IDs {}.".format(
                dataset, missing
            )
        )
    return [sum(values) / len(values) for values in grouped]


class PublicPriorReleaseRiskTracker:
    """Update a public leakage prior using prior DP-release quality only.

    Public Cluster scores provide cross-party susceptibility.  At private
    step t, the score is updated from the quality of the DP gradient released
    at step t and becomes available to the controller only at step t+1.
    """

    def __init__(
        self,
        public_scores,
        ema_beta=0.9,
        quality_ema_beta=0.9,
        quality_power=2.0,
        quality_ratio_min=0.8,
        quality_ratio_max=1.2,
    ):
        if len(public_scores) == 0:
            raise ValueError("At least one public risk score is required.")
        values = [float(value) for value in public_scores]
        if not all(torch.isfinite(torch.tensor(values)).tolist()):
            raise ValueError("Public risk scores must be finite.")
        mean_value = sum(values) / len(values)
        if mean_value <= 1e-12:
            priors = [50.0 for _ in values]
        else:
            # A score of 50 denotes the mean public attack advantage. Unlike
            # min-max scaling, this does not force any party to 0 or 100.
            priors = [100.0 * value / (value + mean_value) for value in values]

        self.num_parties = len(priors)
        self.ema_beta = float(min(max(ema_beta, 0.0), 0.999))
        self.quality_ema_beta = float(min(max(quality_ema_beta, 0.0), 0.999))
        self.quality_power = float(max(quality_power, 0.0))
        self.quality_ratio_min = float(max(quality_ratio_min, 1e-6))
        self.quality_ratio_max = float(max(quality_ratio_max, self.quality_ratio_min))
        self._public_scores = values
        self._priors = priors
        self._scores = list(priors)
        self._qualities = [100.0 for _ in priors]
        self._quality_references = None
        self._initialized = [True for _ in priors]

    def update(self, party_id, released_grad):
        # Compatibility with the geometry tracker call sites.  The public
        # mode updates once per complete party release through update_qualities.
        return self._scores[int(party_id)]

    def update_qualities(self, qualities):
        if len(qualities) != self.num_parties:
            raise ValueError("One release-quality value is required per party.")
        bounded = [
            float(min(max(float(quality), 0.0), 100.0))
            for quality in qualities
        ]
        if self._quality_references is None:
            self._quality_references = list(bounded)
            self._qualities = list(bounded)
            return self.get_scores()

        for party_id, quality in enumerate(bounded):
            reference = self._quality_references[party_id]
            if reference <= 1e-12:
                ratio = 1.0 if quality <= 1e-12 else self.quality_ratio_max
            else:
                ratio = quality / reference
            ratio = min(max(ratio, self.quality_ratio_min), self.quality_ratio_max)

            probability = min(max(self._priors[party_id] / 100.0, 1e-6), 1.0 - 1e-6)
            prior_odds = probability / (1.0 - probability)
            adjusted_odds = prior_odds * (ratio ** self.quality_power)
            target = 100.0 * adjusted_odds / (1.0 + adjusted_odds)
            self._scores[party_id] = float(
                self.ema_beta * self._scores[party_id]
                + (1.0 - self.ema_beta) * target
            )
            self._qualities[party_id] = quality
            self._quality_references[party_id] = float(
                self.quality_ema_beta * reference
                + (1.0 - self.quality_ema_beta) * quality
            )
        return self.get_scores()

    def get_scores(self):
        return list(self._scores)

    def get_initialized(self):
        return list(self._initialized)

    def get_public_priors(self):
        return list(self._priors)

    def get_public_scores(self):
        return list(self._public_scores)

    def get_last_qualities(self):
        return list(self._qualities)


class ReleasedGradientRiskTracker:
    """Estimate relative label-leakage risk from past DP releases only.

    The tracker compares the local-neighborhood concentration of a window of
    released per-sample gradients with a column-shuffled reference. Shuffling
    preserves coordinate-wise marginals (including a fixed Top-K mask) while
    destroying sample-level joint structure. The resulting score is in
    [0, 100] and provides the RL risk state without executing an online attack
    or reading private labels.
    """

    def __init__(
        self,
        num_parties,
        num_classes,
        window_samples=256,
        min_samples=64,
        update_interval=10,
        ema_beta=0.9,
        max_features=256,
        seed=0,
    ):
        self.num_parties = int(num_parties)
        self.num_classes = max(int(num_classes), 2)
        self.window_samples = max(int(window_samples), 2)
        self.min_samples = min(
            self.window_samples,
            max(int(min_samples), 2 * self.num_classes),
        )
        self.update_interval = max(int(update_interval), 1)
        self.ema_beta = float(min(max(ema_beta, 0.0), 0.999))
        self.max_features = max(int(max_features), self.num_classes)
        self.seed = int(seed)

        self._buffers = [None for _ in range(self.num_parties)]
        self._buffer_sizes = [0 for _ in range(self.num_parties)]
        self._write_positions = [0 for _ in range(self.num_parties)]
        self._ordered_scratch = [None for _ in range(self.num_parties)]
        self._updates = [0 for _ in range(self.num_parties)]
        self._scores = [0.0 for _ in range(self.num_parties)]
        self._initialized = [False for _ in range(self.num_parties)]

    @staticmethod
    def _flatten_samples(released_grad):
        if released_grad is None or released_grad.ndim == 0:
            return None
        if released_grad.ndim == 1:
            flat = released_grad.detach().reshape(1, -1)
        else:
            flat = released_grad.detach().reshape(released_grad.shape[0], -1)
        flat = torch.nan_to_num(flat.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return flat.cpu()

    def _reset_ring(self, party_id, flat):
        feature_dim = int(flat.shape[1])
        self._buffers[party_id] = torch.empty(
            (self.window_samples, feature_dim),
            dtype=flat.dtype,
            device="cpu",
        )
        self._ordered_scratch[party_id] = torch.empty_like(self._buffers[party_id])
        self._buffer_sizes[party_id] = 0
        self._write_positions[party_id] = 0

    def _append(self, party_id, flat):
        current = self._buffers[party_id]
        if current is None or current.shape[1] != flat.shape[1]:
            self._reset_ring(party_id, flat)
            current = self._buffers[party_id]

        incoming_count = int(flat.shape[0])
        if incoming_count >= self.window_samples:
            current.copy_(flat[-self.window_samples :])
            self._buffer_sizes[party_id] = self.window_samples
            self._write_positions[party_id] = 0
            return

        write_position = self._write_positions[party_id]
        first_count = min(incoming_count, self.window_samples - write_position)
        current[write_position : write_position + first_count].copy_(flat[:first_count])
        remaining = incoming_count - first_count
        if remaining > 0:
            current[:remaining].copy_(flat[first_count:])

        self._write_positions[party_id] = (write_position + incoming_count) % self.window_samples
        self._buffer_sizes[party_id] = min(
            self.window_samples,
            self._buffer_sizes[party_id] + incoming_count,
        )

    def _ordered_buffer(self, party_id):
        current = self._buffers[party_id]
        if current is None:
            return None

        size = self._buffer_sizes[party_id]
        if size < self.window_samples:
            return current[:size]

        write_position = self._write_positions[party_id]
        if write_position == 0:
            return current

        scratch = self._ordered_scratch[party_id]
        tail_count = self.window_samples - write_position
        scratch[:tail_count].copy_(current[write_position:])
        scratch[tail_count:].copy_(current[:write_position])
        return scratch

    @staticmethod
    def _locality_score(matrix):
        if matrix.shape[0] < 3:
            return 0.0
        distances = torch.cdist(matrix, matrix, p=2)
        diagonal = torch.eye(matrix.shape[0], dtype=torch.bool)
        off_diagonal = distances[~diagonal]
        if off_diagonal.numel() == 0:
            return 0.0
        global_scale = float(torch.median(off_diagonal).item())
        if global_scale <= 1e-12:
            return 0.0
        distances = distances.masked_fill(diagonal, float("inf"))
        nearest = distances.min(dim=1).values
        locality = 1.0 - float(nearest.mean().item()) / global_scale
        return float(min(max(locality, 0.0), 1.0))

    def _compute(self, party_id):
        matrix = self._ordered_buffer(party_id)
        if matrix is None or matrix.shape[0] < self.min_samples:
            return None

        row_norms = torch.linalg.vector_norm(matrix, ord=2, dim=1, keepdim=True)
        nonzero_rows = row_norms.squeeze(1) > 1e-12
        matrix = matrix[nonzero_rows]
        if matrix.shape[0] < self.min_samples:
            return None

        matrix = matrix / torch.clamp(
            torch.linalg.vector_norm(matrix, ord=2, dim=1, keepdim=True),
            min=1e-12,
        )
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        variances = centered.square().mean(dim=0)
        if variances.numel() == 0 or float(variances.max().item()) <= 1e-12:
            return None
        active = variances > max(float(variances.max().item()) * 1e-8, 1e-12)
        matrix = matrix[:, active]
        variances = variances[active]
        if matrix.shape[1] == 0:
            return None

        if matrix.shape[1] > self.max_features:
            keep = torch.topk(variances, k=self.max_features, largest=True).indices
            matrix = matrix[:, keep]

        observed = self._locality_score(matrix)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 104729 * party_id + self._updates[party_id])
        random_keys = torch.rand(matrix.shape, generator=generator)
        permutation = torch.argsort(random_keys, dim=0)
        shuffled = torch.gather(matrix, dim=0, index=permutation)
        shuffled = shuffled / torch.clamp(
            torch.linalg.vector_norm(shuffled, ord=2, dim=1, keepdim=True),
            min=1e-12,
        )
        reference = self._locality_score(shuffled)

        excess = max(observed - reference, 0.0)
        normalized = excess / max(1.0 - reference, 1e-12)
        raw_risk = 100.0 * min(max(normalized, 0.0), 1.0)
        return raw_risk

    def update(self, party_id, released_grad):
        party_id = int(party_id)
        if party_id < 0 or party_id >= self.num_parties:
            raise IndexError("party_id is outside the configured party range")

        flat = self._flatten_samples(released_grad)
        if flat is None or flat.numel() == 0:
            return self._scores[party_id]

        self._append(party_id, flat)
        self._updates[party_id] += 1
        if self._updates[party_id] % self.update_interval != 0:
            return self._scores[party_id]

        result = self._compute(party_id)
        if result is None:
            return self._scores[party_id]

        raw_risk = result
        # Cold-start from zero so the first valid geometry estimate cannot
        # immediately trip the fixed gate on a single early observation.
        score = self.ema_beta * self._scores[party_id] + (1.0 - self.ema_beta) * raw_risk
        self._initialized[party_id] = True
        self._scores[party_id] = float(min(max(score, 0.0), 100.0))
        return self._scores[party_id]

    def get_scores(self):
        return list(self._scores)

    def get_initialized(self):
        return list(self._initialized)
