import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split

from .vflbase import BaseVFL


# Deliberately stronger-than-standard FORA calibration. These are kept local
# so the experimental CLI remains unchanged.
FORA_ORACLE_NUM_SAMPLES = 1000
FORA_ORACLE_DECODER_EPOCHS = 20
FORA_ORACLE_LR = 1e-4


class VectorFIADecoder(nn.Module):
    def __init__(self, input_dim, image_shape):
        super().__init__()
        output_dim = int(np.prod(image_shape))
        hidden_dim = max(512, min(2048, input_dim * 2))
        self.image_shape = tuple(image_shape)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(x.size(0), *self.image_shape)


class ConvFIADecoder(nn.Module):
    def __init__(self, input_shape, output_nc):
        super().__init__()
        input_nc = input_shape[0]
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_nc, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_nc, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class SpatialVectorFIADecoder(nn.Module):
    def __init__(self, input_dim, image_shape):
        super().__init__()
        self.image_shape = tuple(image_shape)
        output_dim = int(np.prod(image_shape))
        if input_dim != output_dim:
            raise ValueError(
                f"SpatialVectorFIADecoder requires input_dim == output_dim, got {input_dim} and {output_dim}"
            )

        channels = self.image_shape[0]
        self.project = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.project(x)
        out = out.view(x.size(0), *self.image_shape)
        return self.refine(out)


class FORASubstituteEncoder(nn.Module):
    """Public-data encoder whose output matches the victim smashed-data shape."""

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.output_shape = tuple(output_shape)
        output_dim = int(np.prod(self.output_shape))
        if len(input_shape) == 3:
            in_channels = int(input_shape[0])
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            feature_dim = 128 * 4 * 4
        else:
            input_dim = int(np.prod(input_shape))
            feature_dim = max(256, min(1024, input_dim * 2))
            self.features = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
            )
        self.project = nn.Linear(feature_dim, output_dim)

    def forward(self, x):
        out = self.project(self.features(x))
        return out.view(x.size(0), *self.output_shape)


class FORAFeatureDiscriminator(nn.Module):
    def __init__(self, feature_shape):
        super().__init__()
        input_dim = int(np.prod(feature_shape))
        hidden_dim = max(128, min(512, input_dim))
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, max(64, hidden_dim // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(max(64, hidden_dim // 2), 1),
        )

    def forward(self, x):
        return self.model(x).flatten()


class Attacker(BaseVFL):
    def __init__(self, args, model, train_dataset, test_dataset):
        super().__init__(args, model, train_dataset, test_dataset)
        self.fia_mode = getattr(args, "fia_mode", "decoder")
        self.fia_has_run_final = False
        self.decoder_model = None
        self.fora_substitute_model = None
        self.fora_discriminator = None
        self.fora_substitute_optimizer = None
        self.fora_discriminator_optimizer = None
        self.fora_public_inputs = None
        self.fora_public_targets = None
        self.fora_online_steps = 0
        self.fora_online_totals = {
            "discriminator": 0.0,
            "adversarial": 0.0,
            "mmd": 0.0,
        }
        self.fora_sampling_generator = torch.Generator().manual_seed(
            int(getattr(args, "completion_aux_seed", 0))
        )
        self.fora_victim_buffer = None
        self.decoder_ckpt_path = os.path.join(self.log_dir, "ressfl_fia_decoder.pt")
        self.fora_ckpt_path = os.path.join(self.log_dir, "fora_models.pt")
        self.fia_res_dir = os.path.join(
            self.output_root,
            "fia_results",
            self.args.dataset,
            self.fia_mode,
        )
        os.makedirs(self.fia_res_dir, exist_ok=True)
        print(f"Attacker: ResSFL FIA (Mode: {self.fia_mode})")

    def _cifar10_stats(self, tensor):
        mean = tensor.new_tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        return mean, std

    def _to_pixel_space(self, tensor):
        if self.args.dataset == "cifar10":
            mean, std = self._cifar10_stats(tensor)
            tensor = tensor * std + mean
        return tensor.clamp(0.0, 1.0)

    def _build_visual_grid(self, image_batch, max_samples, vis_scale, vis_padding):
        vis_batch = image_batch[:max_samples].detach().cpu()
        if vis_scale > 1:
            vis_batch = F.interpolate(vis_batch, scale_factor=vis_scale, mode="nearest")

        vis_grid = torchvision.utils.make_grid(
            vis_batch,
            nrow=max_samples,
            padding=vis_padding,
            pad_value=1.0,
        )
        return vis_grid

    def _save_reconstruction_visuals(self, reconstructed, real_img, prefix, batch_idx, full_img=None):
        if not bool(getattr(self.args, "fia_save_visuals", False)):
            return

        save_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        os.makedirs(self.fia_res_dir, exist_ok=True)

        if getattr(self.args, "adavfed", False):
            defense_tag = "adavfed"
        elif getattr(self.args, "acvfl", False):
            defense_tag = "acvfl"
        elif getattr(self.args, "defense_all", False):
            defense_tag = "defense_all"
        elif getattr(self.args, "ppdl", False):
            defense_tag = "ppdl"
        elif getattr(self.args, "rl", False):
            defense_tag = "rl"
        else:
            defense_tag = "null"

        num_samples = int(getattr(self.args, "fia_save_num_samples", 6))
        if reconstructed.dim() != 4 or real_img.dim() != 4:
            return

        batch_size = min(reconstructed.size(0), real_img.size(0))
        if batch_size <= 0:
            return

        vis_scale = max(1, int(getattr(self.args, "fia_vis_scale", 12)))
        vis_padding = max(0, int(getattr(self.args, "fia_vis_padding", 8)))
        group_size = max(1, num_samples) if num_samples > 0 else min(6, batch_size)
        max_groups = int(getattr(self.args, "fia_save_num_groups", 1))
        num_groups = min(max_groups, int(np.ceil(batch_size / group_size)))

        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = min(start_idx + group_size, batch_size)
            if start_idx >= end_idx:
                break

            recon_slice = reconstructed[start_idx:end_idx]
            real_slice = real_img[start_idx:end_idx]
            full_slice = None if full_img is None else full_img[start_idx:end_idx]
            current_group_size = end_idx - start_idx
            base_name = f"{prefix}_P{self.args.attack_id}_B{batch_idx}_G{group_idx}_{save_tag}_{defense_tag}"

            recon_grid = self._build_visual_grid(
                recon_slice,
                max_samples=current_group_size,
                vis_scale=vis_scale,
                vis_padding=vis_padding,
            )
            real_grid = self._build_visual_grid(
                real_slice,
                max_samples=current_group_size,
                vis_scale=vis_scale,
                vis_padding=vis_padding,
            )

            torchvision.utils.save_image(
                recon_grid,
                os.path.join(self.fia_res_dir, f"{base_name}_recon.jpg"),
            )
            torchvision.utils.save_image(
                real_grid,
                os.path.join(self.fia_res_dir, f"{base_name}_real.jpg"),
            )
            if full_slice is not None and full_slice.dim() == 4:
                full_grid = self._build_visual_grid(
                    full_slice,
                    max_samples=current_group_size,
                    vis_scale=vis_scale,
                    vis_padding=vis_padding,
                )
                torchvision.utils.save_image(
                    full_grid,
                    os.path.join(self.fia_res_dir, f"{base_name}_full.jpg"),
                )

    def _list_data_files(self):
        return sorted(
            [
                file_name
                for file_name in os.listdir(self.data_dir)
                if file_name.endswith(".pt")
            ]
        )

    def _load_batch(self, batch_file):
        data_list, emb_list, _, labels = torch.load(
            os.path.join(self.data_dir, batch_file),
            map_location="cpu",
        )
        pixel_data = [
            self._to_pixel_space(party.float().to(self.device))
            for party in data_list
        ]
        real_img = pixel_data[self.args.attack_id]
        target_emb = emb_list[self.args.attack_id].float().to(self.device)
        full_img = self._recover_full_image(pixel_data)
        return real_img, target_emb, full_img, labels.to(self.device)

    def _recover_full_image(self, data_list):
        if not data_list:
            return None
        if self.args.division_mode != "vertical":
            return None

        sample_tensor = data_list[0]
        if not isinstance(sample_tensor, torch.Tensor) or sample_tensor.dim() != 4:
            return None

        if self.args.dataset == "criteo":
            return None

        try:
            full_img = torch.cat([party.float() for party in data_list], dim=3)
        except RuntimeError:
            return None
        return full_img.to(self.device)

    def _select_data_files(self, data_files):
        max_batches = int(getattr(self.args, "fia_max_batches", -1))
        if max_batches <= 0 or len(data_files) <= max_batches:
            print(f"[ResSFL FIA] Using all {len(data_files)} cached batches.")
            return data_files

        selected_indices = np.linspace(
            0, len(data_files) - 1, num=max_batches, dtype=int
        )
        selected_files = [data_files[idx] for idx in selected_indices]
        print(
            f"[ResSFL FIA] Using {len(selected_files)}/{len(data_files)} cached batches "
            "for attack evaluation."
        )
        return selected_files

    def _build_decoder(self, sample_emb, sample_img):
        if sample_emb.dim() == 2:
            if sample_emb.shape[1] == int(np.prod(sample_img.shape[1:])):
                return SpatialVectorFIADecoder(sample_emb.shape[1], sample_img.shape[1:])
            return VectorFIADecoder(sample_emb.shape[1], sample_img.shape[1:])
        if sample_emb.dim() == 4:
            return ConvFIADecoder(sample_emb.shape[1:], sample_img.shape[1])
        raise ValueError(
            f"Unsupported embedding shape for FIA decoder: {tuple(sample_emb.shape)}"
        )

    def _build_decoder_dataloaders(self, data_files):
        img_list = []
        emb_list = []
        for batch_file in data_files:
            imgs, embs, _, _ = self._load_batch(batch_file)
            img_list.append(imgs.detach().cpu())
            emb_list.append(embs.detach().cpu())

        imgs_all = torch.cat(img_list, dim=0).float()
        embs_all = torch.cat(emb_list, dim=0).float()
        dataset = TensorDataset(embs_all, imgs_all)

        if len(dataset) < 2:
            raise RuntimeError("Decoder FIA requires at least two samples to create a train/validation split.")

        val_ratio = float(getattr(self.args, "fia_decoder_val_ratio", 0.2))
        val_size = max(1, int(len(dataset) * val_ratio))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            train_size = len(dataset) - 1
            val_size = 1

        split_seed = int(getattr(self.args, "completion_aux_seed", 0))
        split_generator = torch.Generator().manual_seed(split_seed)
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=split_generator,
        )

        batch_size = max(1, int(getattr(self.args, "fia_decoder_batch_size", 32)))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, imgs_all[0:1], embs_all[0:1]

    @staticmethod
    def _mk_mmd(source, target):
        source = source.flatten(start_dim=1)
        target = target.flatten(start_dim=1)
        combined = torch.cat([source, target], dim=0)
        distances = torch.cdist(combined, combined).pow(2)
        positive = distances.detach()[distances.detach() > 0]
        bandwidth = positive.median() if positive.numel() else distances.new_tensor(1.0)
        bandwidth = bandwidth.clamp_min(1e-6)

        kernel = distances.new_zeros(distances.shape)
        for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
            kernel = kernel + torch.exp(-distances / (2.0 * bandwidth * scale))

        source_size = source.size(0)
        k_ss = kernel[:source_size, :source_size]
        k_tt = kernel[source_size:, source_size:]
        k_st = kernel[:source_size, source_size:]
        return k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()

    def _collect_fora_public_data(self):
        if not self.aux_val_dataset:
            raise RuntimeError(
                "FORA requires a public auxiliary dataset. Use --aux_public_dataset auto "
                "or choose an available public dataset."
            )

        max_samples = max(2, int(getattr(self.args, "fora_aux_num_samples", 5000)))
        model_inputs = []
        pixel_targets = []
        collected = 0
        for data, _ in self.aux_val_dataset:
            party_input = data[self.args.attack_id].float()
            remaining = max_samples - collected
            if remaining <= 0:
                break
            party_input = party_input[:remaining]
            model_inputs.append(party_input.cpu())
            pixel_targets.append(self._to_pixel_space(party_input.to(self.device)).cpu())
            collected += party_input.size(0)

        if collected < 2:
            raise RuntimeError("FORA requires at least two public auxiliary samples.")
        return torch.cat(model_inputs), torch.cat(pixel_targets)

    def _collect_fora_victim_embeddings(self, data_files):
        embeddings = []
        for batch_file in data_files:
            _, emb_list, _, _ = torch.load(
                os.path.join(self.data_dir, batch_file),
                map_location="cpu",
            )
            embeddings.append(emb_list[self.args.attack_id].float())
        return torch.cat(embeddings)

    def _build_fora_oracle_loader(self, data_files):
        embeddings = []
        targets = []
        remaining = FORA_ORACLE_NUM_SAMPLES
        for batch_file in data_files:
            if remaining <= 0:
                break
            real_img, target_emb, _, _ = self._load_batch(batch_file)
            take = min(remaining, target_emb.size(0))
            embeddings.append(target_emb[:take].detach().cpu())
            targets.append(real_img[:take].detach().cpu())
            remaining -= take

        dataset = TensorDataset(torch.cat(embeddings), torch.cat(targets))
        generator = torch.Generator().manual_seed(
            int(getattr(self.args, "completion_aux_seed", 0)) + 1701
        )
        return DataLoader(
            dataset,
            batch_size=min(32, len(dataset)),
            shuffle=True,
            generator=generator,
        )

    def _calibrate_fora_decoder_with_oracle_pairs(self, data_files):
        oracle_loader = self._build_fora_oracle_loader(data_files)
        optimizer = torch.optim.Adam(self.decoder_model.parameters(), lr=FORA_ORACLE_LR)
        print(
            "[FORA Oracle Calibration] samples={} | epochs={} | lr={}".format(
                len(oracle_loader.dataset),
                FORA_ORACLE_DECODER_EPOCHS,
                FORA_ORACLE_LR,
            )
        )
        for epoch in range(FORA_ORACLE_DECODER_EPOCHS):
            self.decoder_model.train()
            mse_total = 0.0
            steps = 0
            for victim_embedding, target in oracle_loader:
                victim_embedding = victim_embedding.to(self.device)
                target = target.to(self.device)
                reconstructed = self.decoder_model(victim_embedding)
                loss, mse_loss = self._decoder_training_loss(reconstructed, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mse_total += mse_loss.item()
                steps += 1
            print(
                "[FORA Oracle Calibration] Epoch {}/{} | Victim-pair MSE={:.6f}".format(
                    epoch + 1,
                    FORA_ORACLE_DECODER_EPOCHS,
                    mse_total / max(1, steps),
                )
            )

    def _decoder_training_loss(self, reconstructed, targets):
        mse_loss = F.mse_loss(reconstructed, targets)
        loss = mse_loss
        if self.args.dataset == "cifar10":
            ssim_value = self._ssim_per_sample(reconstructed, targets).mean()
            reconstructed_dx = reconstructed[:, :, :, 1:] - reconstructed[:, :, :, :-1]
            targets_dx = targets[:, :, :, 1:] - targets[:, :, :, :-1]
            reconstructed_dy = reconstructed[:, :, 1:, :] - reconstructed[:, :, :-1, :]
            targets_dy = targets[:, :, 1:, :] - targets[:, :, :-1, :]
            gradient_loss = (
                F.l1_loss(reconstructed_dx, targets_dx)
                + F.l1_loss(reconstructed_dy, targets_dy)
            )
            loss = loss + 0.1 * (1.0 - ssim_value) + 0.02 * gradient_loss
        return loss, mse_loss

    @staticmethod
    def _ssim_per_sample(reconstructed, real_img):
        """Compute ResSFL's Gaussian-window SSIM for each image in a batch."""
        if reconstructed.dim() != 4 or real_img.dim() != 4:
            raise ValueError("FIA SSIM requires image tensors with shape [N, C, H, W].")

        window_size = 11
        sigma = 1.5
        coordinates = torch.arange(
            window_size,
            dtype=reconstructed.dtype,
            device=reconstructed.device,
        )
        gaussian_1d = torch.exp(
            -((coordinates - window_size // 2) ** 2) / (2 * sigma ** 2)
        )
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] @ gaussian_1d[None, :]
        channels = reconstructed.shape[1]
        window = gaussian_2d.expand(channels, 1, window_size, window_size).contiguous()
        padding = window_size // 2

        mean_recon = F.conv2d(reconstructed, window, padding=padding, groups=channels)
        mean_real = F.conv2d(real_img, window, padding=padding, groups=channels)
        var_recon = F.conv2d(
            reconstructed * reconstructed,
            window,
            padding=padding,
            groups=channels,
        ) - mean_recon.pow(2)
        var_real = F.conv2d(
            real_img * real_img,
            window,
            padding=padding,
            groups=channels,
        ) - mean_real.pow(2)
        covariance = F.conv2d(
            reconstructed * real_img,
            window,
            padding=padding,
            groups=channels,
        ) - mean_recon * mean_real

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mean_recon * mean_real + c1) * (2 * covariance + c2)) / (
            (mean_recon.pow(2) + mean_real.pow(2) + c1) * (var_recon + var_real + c2)
        )
        return ssim_map.flatten(start_dim=1).mean(dim=1)

    def _collect_metrics(self, reconstructed, real_img):
        if reconstructed.shape != real_img.shape:
            raise ValueError(
                "FIA reconstruction and target shapes must match, got "
                f"{tuple(reconstructed.shape)} and {tuple(real_img.shape)}."
            )
        mse_per_sample = (reconstructed - real_img).pow(2).flatten(start_dim=1).mean(dim=1)
        ssim_per_sample = self._ssim_per_sample(reconstructed, real_img)
        return mse_per_sample.detach().cpu(), ssim_per_sample.detach().cpu()

    def _record_fia_metrics(self, mse_per_sample, ssim_per_sample):
        mse_values = torch.cat(mse_per_sample).numpy()
        ssim_values = torch.cat(ssim_per_sample).numpy()
        metrics_summary = {
            "mode": self.fia_mode,
            "num_samples": int(mse_values.size),
            "mse_mean": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values, ddof=1)) if mse_values.size > 1 else 0.0,
            "ssim_mean": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values, ddof=1)) if ssim_values.size > 1 else 0.0,
        }
        print(
            "[FIA Result][{}] samples={} | MSE={:.6f}+/-{:.6f} | SSIM={:.6f}+/-{:.6f}".format(
                self.fia_mode.capitalize(),
                metrics_summary["num_samples"],
                metrics_summary["mse_mean"],
                metrics_summary["mse_std"],
                metrics_summary["ssim_mean"],
                metrics_summary["ssim_std"],
            )
        )
        self.metrics.attack_mse.append(metrics_summary["mse_mean"])
        self.metrics.attack_ssim.append(metrics_summary["ssim_mean"])
        self.metrics.fia_sample_metrics.append(metrics_summary)
        self.metrics.write()
        return metrics_summary

    def attack(self, init=False):
        data_files = self._list_data_files()
        if not data_files:
            print("[ResSFL FIA] No cached attack data found, skipping.")
            return
        data_files = self._select_data_files(data_files)

        if self.fia_mode == "decoder":
            if init or self.decoder_model is None:
                self._train_decoder(data_files)
            self._run_decoder_fia(data_files)
        elif self.fia_mode == "fora":
            if init or self.decoder_model is None or self.fora_substitute_model is None:
                self._train_fora(data_files)
            self._run_fora(data_files)
        else:
            raise ValueError(f"Unsupported fia_mode: {self.fia_mode}")
        self.fia_has_run_final = True

    def _initialize_fora_alignment(self, public_input_shape, victim_embedding_shape):
        if self.fora_substitute_model is not None:
            if tuple(self.fora_substitute_model.output_shape) != tuple(victim_embedding_shape):
                raise RuntimeError(
                    "FORA victim embedding shape changed during training: {} -> {}.".format(
                        self.fora_substitute_model.output_shape,
                        tuple(victim_embedding_shape),
                    )
                )
            return

        self.fora_substitute_model = FORASubstituteEncoder(
            public_input_shape,
            victim_embedding_shape,
        ).to(self.device)
        self.fora_discriminator = FORAFeatureDiscriminator(victim_embedding_shape).to(self.device)
        lr = float(getattr(self.args, "fora_lr", 1e-3))
        self.fora_substitute_optimizer = torch.optim.Adam(
            self.fora_substitute_model.parameters(),
            lr=lr,
        )
        self.fora_discriminator_optimizer = torch.optim.Adam(
            self.fora_discriminator.parameters(),
            lr=lr * float(getattr(self.args, "fora_discriminator_lr_ratio", 0.25)),
        )
        print(
            "[FORA] Online alignment initialized | public_input={} | victim_embedding={}".format(
                tuple(public_input_shape),
                tuple(victim_embedding_shape),
            )
        )

    def _fora_alignment_update(self, public_batch, victim_batch):
        public_batch = public_batch.to(self.device)
        victim_batch = victim_batch.detach().to(self.device)
        bce = F.binary_cross_entropy_with_logits

        # A shared victim-domain transform stabilizes adversarial training while
        # preserving the affine mismatch that the substitute must learn.
        reduce_dims = (0,)
        victim_mean = victim_batch.mean(dim=reduce_dims, keepdim=True)
        victim_std = victim_batch.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1e-4)

        def normalize(features):
            return (features - victim_mean) / victim_std

        self.fora_discriminator_optimizer.zero_grad()
        with torch.no_grad():
            public_features = self.fora_substitute_model(public_batch)
        normalized_victim = normalize(victim_batch)
        victim_logits = self.fora_discriminator(normalized_victim)
        public_logits = self.fora_discriminator(normalize(public_features.detach()))
        discriminator_loss = (
            bce(victim_logits, torch.ones_like(victim_logits))
            + bce(public_logits, torch.zeros_like(public_logits))
        )
        discriminator_loss.backward()
        self.fora_discriminator_optimizer.step()

        for parameter in self.fora_discriminator.parameters():
            parameter.requires_grad_(False)
        self.fora_substitute_optimizer.zero_grad()
        public_features = self.fora_substitute_model(public_batch)
        normalized_public = normalize(public_features)
        public_logits = self.fora_discriminator(normalized_public)
        adversarial_loss = bce(public_logits, torch.ones_like(public_logits))
        mmd_loss = self._mk_mmd(normalized_public, normalized_victim)
        substitute_loss = (
            float(getattr(self.args, "fora_adv_weight", 1.0)) * adversarial_loss
            + float(getattr(self.args, "fora_mmd_weight", 1.0)) * mmd_loss
        )
        substitute_loss.backward()
        self.fora_substitute_optimizer.step()
        for parameter in self.fora_discriminator.parameters():
            parameter.requires_grad_(True)

        return discriminator_loss.item(), adversarial_loss.item(), mmd_loss.item()

    def fora_online_alignment_step(self, victim_embedding):
        if self.fora_public_inputs is None:
            self.fora_public_inputs, self.fora_public_targets = self._collect_fora_public_data()
            print(
                f"[FORA] Online public pool prepared: {len(self.fora_public_inputs)} samples; "
                "victim raw images are excluded from alignment."
            )

        self._initialize_fora_alignment(
            self.fora_public_inputs.shape[1:],
            victim_embedding.shape[1:],
        )
        buffer_limit = max(2, int(getattr(self.args, "fora_victim_buffer_size", 1024)))
        incoming = victim_embedding.detach().cpu()
        if self.fora_victim_buffer is None:
            self.fora_victim_buffer = incoming
        else:
            self.fora_victim_buffer = torch.cat((self.fora_victim_buffer, incoming), dim=0)[-buffer_limit:]

        current_size = min(
            max(2, int(getattr(self.args, "fora_batch_size", 64))),
            len(self.fora_victim_buffer),
            len(self.fora_public_inputs),
        )
        if current_size < 2:
            return
        update_count = max(1, int(getattr(self.args, "fora_online_updates", 3)))
        losses = []
        for _ in range(update_count):
            public_indices = torch.randint(
                len(self.fora_public_inputs), (current_size,), generator=self.fora_sampling_generator
            )
            victim_indices = torch.randint(
                len(self.fora_victim_buffer), (current_size,), generator=self.fora_sampling_generator
            )
            losses.append(
                self._fora_alignment_update(
                    self.fora_public_inputs.index_select(0, public_indices),
                    self.fora_victim_buffer.index_select(0, victim_indices),
                )
            )
        discriminator_loss = sum(item[0] for item in losses) / update_count
        adversarial_loss = sum(item[1] for item in losses) / update_count
        mmd_loss = sum(item[2] for item in losses) / update_count
        self.fora_online_steps += 1
        self.fora_online_totals["discriminator"] += discriminator_loss
        self.fora_online_totals["adversarial"] += adversarial_loss
        self.fora_online_totals["mmd"] += mmd_loss

        log_interval = max(100, int(getattr(self.args, "fia_progress_interval", 10)) * 10)
        if self.fora_online_steps == 1 or self.fora_online_steps % log_interval == 0:
            print(
                "[FORA][Online S{}] D={:.6f} | Adv={:.6f} | MK-MMD={:.6f}".format(
                    self.fora_online_steps,
                    discriminator_loss,
                    adversarial_loss,
                    mmd_loss,
                )
            )

    def _train_fora(self, data_files):
        print("[FORA] Finalizing the online substitute model and training the inverse decoder...")
        if self.fora_public_inputs is None:
            self.fora_public_inputs, self.fora_public_targets = self._collect_fora_public_data()
        public_inputs = self.fora_public_inputs
        public_targets = self.fora_public_targets
        victim_embeddings = self._collect_fora_victim_embeddings(data_files)
        self._initialize_fora_alignment(
            public_inputs.shape[1:],
            victim_embeddings.shape[1:],
        )
        batch_size = max(2, int(getattr(self.args, "fora_batch_size", 64)))
        public_loader = DataLoader(
            TensorDataset(public_inputs, public_targets),
            batch_size=batch_size,
            shuffle=True,
        )
        lr = float(getattr(self.args, "fora_lr", 1e-3))
        if self.fora_online_steps == 0:
            print(
                "[FORA] Warning: no online alignment steps were observed; "
                "using final snapshots as a compatibility fallback."
            )
            alignment_epochs = max(1, int(getattr(self.args, "fora_alignment_epochs", 50)))
            victim_loader = DataLoader(TensorDataset(victim_embeddings), batch_size=batch_size, shuffle=True)
            for _ in range(alignment_epochs):
                victim_iterator = iter(victim_loader)
                for public_batch, _ in public_loader:
                    try:
                        victim_batch = next(victim_iterator)[0]
                    except StopIteration:
                        victim_iterator = iter(victim_loader)
                        victim_batch = next(victim_iterator)[0]
                    current_size = min(public_batch.size(0), victim_batch.size(0))
                    if current_size >= 2:
                        losses = self._fora_alignment_update(
                            public_batch[:current_size],
                            victim_batch[:current_size],
                        )
                        self.fora_online_steps += 1
                        for key, value in zip(
                            ("discriminator", "adversarial", "mmd"),
                            losses,
                        ):
                            self.fora_online_totals[key] += value

        print(
            "[FORA] Alignment completed over {} VFL steps | D={:.6f} | Adv={:.6f} | MK-MMD={:.6f}".format(
                self.fora_online_steps,
                self.fora_online_totals["discriminator"] / max(1, self.fora_online_steps),
                self.fora_online_totals["adversarial"] / max(1, self.fora_online_steps),
                self.fora_online_totals["mmd"] / max(1, self.fora_online_steps),
            )
        )

        self.fora_substitute_model.eval()
        for parameter in self.fora_substitute_model.parameters():
            parameter.requires_grad_(False)

        sample_public = public_inputs[0:1].to(self.device)
        with torch.no_grad():
            sample_embedding = self.fora_substitute_model(sample_public)
        sample_target = public_targets[0:1].to(self.device)
        self.decoder_model = self._build_decoder(sample_embedding, sample_target).to(self.device)
        decoder_optimizer = torch.optim.Adam(self.decoder_model.parameters(), lr=lr)
        decoder_epochs = max(1, int(getattr(self.args, "fora_decoder_epochs", 50)))
        decoder_log_interval = max(1, decoder_epochs // 10)

        for epoch in range(decoder_epochs):
            mse_total = 0.0
            steps = 0
            self.decoder_model.train()
            for public_batch, target_batch in public_loader:
                public_batch = public_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                with torch.no_grad():
                    public_features = self.fora_substitute_model(public_batch)
                reconstructed = self.decoder_model(public_features)
                loss, mse_loss = self._decoder_training_loss(reconstructed, target_batch)
                decoder_optimizer.zero_grad()
                loss.backward()
                decoder_optimizer.step()
                mse_total += mse_loss.item()
                steps += 1
            if (epoch + 1) == 1 or (epoch + 1) % decoder_log_interval == 0 or (epoch + 1) == decoder_epochs:
                print(
                    f"[FORA] Decoder Epoch {epoch + 1}/{decoder_epochs} | "
                    f"Public Train MSE={mse_total / max(1, steps):.6f}"
                )

        self._calibrate_fora_decoder_with_oracle_pairs(data_files)

        torch.save(
            {
                "substitute": self.fora_substitute_model.state_dict(),
                "decoder": self.decoder_model.state_dict(),
                "public_input_shape": tuple(public_inputs.shape[1:]),
                "victim_embedding_shape": tuple(victim_embeddings.shape[1:]),
                "target_shape": tuple(public_targets.shape[1:]),
            },
            self.fora_ckpt_path,
        )
        print(f"[FORA] Models saved to {self.fora_ckpt_path}")

    def _run_fora(self, data_files):
        print(f"[FORA] Reconstructing {len(data_files)} cached victim batches...")
        all_mse = []
        all_ssim = []
        self.decoder_model.eval()
        progress_interval = max(1, int(getattr(self.args, "fia_progress_interval", 10)))
        with torch.no_grad():
            for batch_idx, batch_file in enumerate(data_files):
                real_img, target_emb, full_img, _ = self._load_batch(batch_file)
                reconstructed = self.decoder_model(target_emb)
                mse_values, ssim_values = self._collect_metrics(reconstructed, real_img)
                all_mse.append(mse_values)
                all_ssim.append(ssim_values)
                if batch_idx == 0:
                    self._save_reconstruction_visuals(
                        reconstructed,
                        real_img,
                        prefix="fia_fora",
                        batch_idx=batch_idx,
                        full_img=full_img,
                    )
                if (
                    (batch_idx + 1) % progress_interval == 0
                    or (batch_idx + 1) == len(data_files)
                ):
                    print(
                        f"[FORA] Progress {batch_idx + 1}/{len(data_files)} | "
                        f"Running Avg MSE: {float(torch.cat(all_mse).mean()):.6f}"
                    )
        self._record_fia_metrics(all_mse, all_ssim)

    def _train_decoder(self, data_files):
        print(f"[ResSFL FIA] Training Decoder on {len(data_files)} cached batches...")
        train_loader, val_loader, sample_img, sample_emb = self._build_decoder_dataloaders(data_files)
        self.decoder_model = self._build_decoder(sample_emb, sample_img).to(self.device)
        print(
            "[ResSFL FIA] Decoder architecture: {} | embedding={} | target={}".format(
                self.decoder_model.__class__.__name__,
                tuple(sample_emb.shape[1:]),
                tuple(sample_img.shape[1:]),
            )
        )

        optimizer = torch.optim.Adam(
            self.decoder_model.parameters(),
            lr=float(getattr(self.args, "fia_decoder_lr", getattr(self.args, "lr_attack_model", 1e-3))),
        )
        criterion = nn.MSELoss().to(self.device)
        num_epochs = max(1, int(getattr(self.args, "fia_decoder_epochs", getattr(self.args, "attack_model_epochs", 10))))
        best_val_loss = float("inf")
        best_state_dict = None

        for epoch in range(num_epochs):
            train_loss_total = 0.0
            train_batches = 0
            self.decoder_model.train()
            for embs, imgs in train_loader:
                embs = embs.to(self.device)
                imgs = imgs.to(self.device)
                out = self.decoder_model(embs)
                loss, mse_loss = self._decoder_training_loss(out, imgs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_total += mse_loss.item()
                train_batches += 1

            avg_train_loss = train_loss_total / max(1, train_batches)

            self.decoder_model.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for embs, imgs in val_loader:
                    embs = embs.to(self.device)
                    imgs = imgs.to(self.device)
                    out = self.decoder_model(embs)
                    val_loss_total += criterion(out, imgs).item()
                    val_batches += 1

            avg_val_loss = val_loss_total / max(1, val_batches)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.decoder_model.state_dict().items()
                }

            print(
                f"[ResSFL FIA] Decoder Epoch {epoch + 1}/{num_epochs} "
                f"Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}"
            )

        if best_state_dict is not None:
            self.decoder_model.load_state_dict(best_state_dict)
            torch.save(best_state_dict, self.decoder_ckpt_path)
        else:
            torch.save(self.decoder_model.state_dict(), self.decoder_ckpt_path)
        print(f"[ResSFL FIA] Decoder checkpoint saved to {self.decoder_ckpt_path}")

    def _run_decoder_fia(self, data_files):
        if self.decoder_model is None:
            sample_img, sample_emb, _, _ = self._load_batch(data_files[0])
            self.decoder_model = self._build_decoder(sample_emb, sample_img).to(self.device)
            if os.path.exists(self.decoder_ckpt_path):
                state_dict = torch.load(self.decoder_ckpt_path, map_location=self.device)
                self.decoder_model.load_state_dict(state_dict)
            else:
                raise RuntimeError(
                    "Decoder FIA requested before decoder training completed."
                )

        print(
            f"[ResSFL FIA] Decoder-based Inference Running on {len(data_files)} cached batches..."
        )
        all_mse = []
        all_ssim = []
        self.decoder_model.eval()
        progress_interval = max(1, int(getattr(self.args, "fia_progress_interval", 10)))

        with torch.no_grad():
            for batch_idx, batch_file in enumerate(data_files):
                real_img, target_emb, full_img, _ = self._load_batch(batch_file)
                reconstructed = self.decoder_model(target_emb)

                mse_values, ssim_values = self._collect_metrics(reconstructed, real_img)
                all_mse.append(mse_values)
                all_ssim.append(ssim_values)

                if batch_idx == 0:
                    self._save_reconstruction_visuals(
                        reconstructed,
                        real_img,
                        prefix="fia_decoder",
                        batch_idx=batch_idx,
                        full_img=full_img,
                    )

                if (
                    (batch_idx + 1) % progress_interval == 0
                    or (batch_idx + 1) == len(data_files)
                ):
                    avg_mse = float(torch.cat(all_mse).mean().item())
                    print(
                        f"[ResSFL FIA] Progress {batch_idx + 1}/{len(data_files)} "
                        f"| Running Avg MSE: {avg_mse:.6f}"
                    )

        self._record_fia_metrics(all_mse, all_ssim)
