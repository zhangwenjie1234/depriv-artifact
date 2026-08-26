"""Command-line configuration and runtime helpers for FIA experiments."""

def add_fia_arguments(parser, hidden_help):
    group = parser.add_argument_group("FIA")
    group.add_argument('--fia_mode', default='decoder', choices=['decoder', 'fora'], help='FIA attack mode')
    group.add_argument('--fia_preset', default='custom', choices=['custom', 'fast', 'paper', 'thorough'],
                       help='high-level FIA preset; use custom to control low-level FIA hyperparameters manually')
    group.add_argument('--fia_max_batches', default=256, type=int, help=hidden_help)
    group.add_argument('--fia_progress_interval', default=10, type=int, help=hidden_help)
    group.add_argument('--fia_save_visuals', action='store_true', default=False,
                       help='save FIA real/reconstruction/full image grids; disabled by default')
    group.add_argument('--fia_disable_cifar_geometric_augmentation', action='store_true', default=False,
                       help='disable CIFAR-10 random crop/flip for FIA experiments; disabled by default')
    group.add_argument('--fia_save_num_samples', default=0, type=int, help=hidden_help)
    group.add_argument('--fia_save_num_groups', default=1, type=int, help=hidden_help)
    group.add_argument('--fia_vis_scale', default=12, type=int, help=hidden_help)
    group.add_argument('--fia_vis_padding', default=8, type=int, help=hidden_help)
    group.add_argument('--fia_run_final_only', default=1, type=int, choices=[0, 1], help=hidden_help)
    group.add_argument('--fia_decoder_lr', default=0.001, type=float, help=hidden_help)
    group.add_argument('--fia_decoder_epochs', default=40, type=int, help=hidden_help)
    group.add_argument('--fia_decoder_batch_size', default=32, type=int, help=hidden_help)
    group.add_argument('--fia_decoder_val_ratio', default=0.2, type=float, help=hidden_help)
    group.add_argument('--fora_aux_num_samples', default=5000, type=int, help=hidden_help)
    group.add_argument('--fora_alignment_epochs', default=50, type=int, help=hidden_help)
    group.add_argument('--fora_decoder_epochs', default=50, type=int, help=hidden_help)
    group.add_argument('--fora_batch_size', default=64, type=int, help=hidden_help)
    group.add_argument('--fora_lr', default=0.001, type=float, help=hidden_help)
    group.add_argument('--fora_discriminator_lr_ratio', default=0.25, type=float, help=hidden_help)
    group.add_argument('--fora_online_updates', default=3, type=int, help=hidden_help)
    group.add_argument('--fora_victim_buffer_size', default=1024, type=int, help=hidden_help)
    group.add_argument('--fora_adv_weight', default=1.0, type=float, help=hidden_help)
    group.add_argument('--fora_mmd_weight', default=1.0, type=float, help=hidden_help)


def validate_fia_args(args):
    if args.fia_max_batches == 0:
        args.fia_max_batches = -1
    if args.fia_max_batches < -1:
        raise ValueError('--fia_max_batches should be -1, 0, or a positive integer.')
    if args.fia_progress_interval <= 0:
        raise ValueError('--fia_progress_interval should be a positive integer.')
    if args.fia_save_num_samples < 0:
        raise ValueError('--fia_save_num_samples should be non-negative.')
    if args.fia_save_num_groups <= 0:
        raise ValueError('--fia_save_num_groups should be a positive integer.')
    if args.fia_vis_scale <= 0:
        raise ValueError('--fia_vis_scale should be a positive integer.')
    if args.fia_vis_padding < 0:
        raise ValueError('--fia_vis_padding should be non-negative.')
    if args.fia_decoder_epochs <= 0:
        raise ValueError('--fia_decoder_epochs should be a positive integer.')
    if args.fia_decoder_batch_size <= 0:
        raise ValueError('--fia_decoder_batch_size should be a positive integer.')
    if not (0.0 < args.fia_decoder_val_ratio < 0.5):
        raise ValueError('--fia_decoder_val_ratio should be in the interval (0, 0.5).')
    if args.fora_aux_num_samples < 2:
        raise ValueError('--fora_aux_num_samples should be at least 2.')
    if args.fora_alignment_epochs <= 0 or args.fora_decoder_epochs <= 0:
        raise ValueError('--fora_alignment_epochs and --fora_decoder_epochs should be positive.')
    if args.fora_batch_size < 2:
        raise ValueError('--fora_batch_size should be at least 2.')
    if args.fora_lr <= 0.0:
        raise ValueError('--fora_lr should be positive.')
    if args.fora_discriminator_lr_ratio <= 0.0:
        raise ValueError('--fora_discriminator_lr_ratio should be positive.')
    if args.fora_online_updates <= 0 or args.fora_victim_buffer_size < 2:
        raise ValueError('--fora_online_updates must be positive and --fora_victim_buffer_size at least 2.')
    if args.fora_adv_weight < 0.0 or args.fora_mmd_weight < 0.0:
        raise ValueError('--fora_adv_weight and --fora_mmd_weight should be non-negative.')


def apply_fia_preset(args, explicitly_provided):
    if args.attack != 'ressfl_fia' or args.fia_preset == 'custom':
        return

    preset_values = {
        'fast': {
            'fia_max_batches': 32, 'fia_decoder_epochs': 20, 'fia_decoder_lr': 0.001,
            'fia_save_num_samples': 0, 'fia_save_num_groups': 2, 'fia_vis_scale': 12,
            'fia_vis_padding': 8, 'fora_alignment_epochs': 10, 'fora_decoder_epochs': 20,
            'fora_aux_num_samples': 1000,
        },
        'paper': {
            'fia_max_batches': 128, 'fia_decoder_epochs': 30, 'fia_decoder_lr': 0.001,
            'fia_save_num_samples': 6, 'fia_save_num_groups': 4, 'fia_vis_scale': 12,
            'fia_vis_padding': 8, 'fora_alignment_epochs': 50, 'fora_decoder_epochs': 50,
            'fora_aux_num_samples': 5000,
        },
        'thorough': {
            'fia_max_batches': 256, 'fia_decoder_epochs': 40, 'fia_decoder_lr': 0.001,
            'fia_save_num_samples': 6, 'fia_save_num_groups': 6, 'fia_vis_scale': 12,
            'fia_vis_padding': 8, 'fora_alignment_epochs': 80, 'fora_decoder_epochs': 80,
            'fora_aux_num_samples': 10000,
        },
    }
    for key, value in preset_values[args.fia_preset].items():
        if not explicitly_provided('--{}'.format(key)):
            setattr(args, key, value)


def needs_fia_public_data(args):
    return bool(getattr(args, 'rl', False)) or (
        args.attack == 'ressfl_fia' and args.fia_mode == 'fora'
    )


def print_fia_config(args):
    if args.attack != 'ressfl_fia':
        return
    if args.fia_mode == 'decoder':
        print(
            "FIA Config -> mode={}, preset={}, attack_id={}, batches={}, epochs={}, lr={}, groups={}".format(
                args.fia_mode, args.fia_preset, args.attack_id, args.fia_max_batches,
                args.fia_decoder_epochs, args.fia_decoder_lr, args.fia_save_num_groups,
            )
        )
    else:
        print(
            "FIA Config -> mode={}, preset={}, attack_id={}, batches={}, aux={}, align_epochs={}, "
            "decoder_epochs={}, lr={}, adv_weight={}, mmd_weight={}, groups={}".format(
                args.fia_mode, args.fia_preset, args.attack_id, args.fia_max_batches,
                args.aux_public_dataset, args.fora_alignment_epochs, args.fora_decoder_epochs,
                args.fora_lr, args.fora_adv_weight, args.fora_mmd_weight, args.fia_save_num_groups,
            )
        )
