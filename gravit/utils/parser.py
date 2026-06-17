import yaml
import argparse

from gravit.utils.cfg_defaults import merge_defaults


def get_args():
    """
    Get the command-line arguments for the configuration
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',           type=str,   help='Path to the configuration file', required=True)

    # Additional arguments from the command line override the configuration from cfg file

    # Root directories for the training process
    parser.add_argument('--root_data',     type=str,   help='Root directory to the data', default='./data')
    parser.add_argument('--root_result',   type=str,   help='Root directory to output', default='./results')

    # Names required for the training process
    parser.add_argument('--exp_name',      type=str,   help='Name of the experiment')
    parser.add_argument('--model_name',    type=str,   help='Name of the model')
    parser.add_argument('--graph_name',    type=str,   help='Name of the graphs')
    parser.add_argument('--loss_name',     type=str,   help='Name of the loss function')
    parser.add_argument('--eval_type',     type=str,   help='Type of the evaluation')

    # Other hyper-parameters
    parser.add_argument('--use_spf',       type=bool,  help='Whether to use the spatial features')
    parser.add_argument('--use_ref',       type=bool,  help='Whether to use the iterative refinement')
    parser.add_argument('--w_ref',         type=float, help='Weight for the iterative refinement')
    parser.add_argument('--num_modality',  type=int,   help='Number of input modalities')
    parser.add_argument('--channel1',      type=int,   help='Filter dimension of the first GCN layers')
    parser.add_argument('--channel2',      type=int,   help='Filter dimension of the rest GCN layers')
    parser.add_argument('--proj_dim',      type=int,   help='Dimension of the projected spatial feature')
    parser.add_argument('--final_dim',     type=int,   help='Dimension of the final output')
    parser.add_argument('--num_att_heads', type=int,   help='Number of attention heads of GATv2')
    parser.add_argument('--dropout',       type=float, help='Dropout for the last GCN layers')
    parser.add_argument('--lr',            type=float, help='Initial learning rate')
    parser.add_argument('--wd',            type=float, help='Weight decay value for regularization')
    parser.add_argument('--batch_size',    type=int,   help='Batch size during the training process')
    parser.add_argument('--sch_param',     type=int,   help='Parameter for lr_scheduler')
    parser.add_argument('--num_epoch',     type=int,   help='Total number of epochs')
    parser.add_argument('--sample_rate',   type=int,   help='Downsampling rate for the input')
    parser.add_argument('--split',         type=int,   help='Which fold to use for cross-validation')
    parser.add_argument(
        '--all_splits',
        action='store_true',
        help='Train folds 1..5 sequentially (SumMe/TVSum 5-fold). Do not pass --split together.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Override cfg device, e.g. cuda:0 or cpu (OOM / driver issues).",
    )
    parser.add_argument(
        '--no_amp',
        action='store_true',
        help='Disable CUDA fp16 autocast: uses more VRAM but avoids some FP16+scatter errors on Windows.',
    )

    return parser.parse_args()


def get_cfg(args):
    """
    Initialize the configuration given the optional command-line arguments
    """

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        delattr(args, 'cfg')

    skip = {"all_splits", "no_amp", "device"}
    for k, v in vars(args).items():
        if k in skip:
            continue
        if v is None and k in cfg:
            continue

        cfg[k] = v

    if getattr(args, "no_amp", False):
        cfg["use_amp"] = False
    if getattr(args, "device", None) is not None:
        cfg["device"] = args.device

    merge_defaults(cfg)
    return cfg
