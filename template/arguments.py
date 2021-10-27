# coding=utf-8
import argparse
import os
from template.tools.logger import get_logger

_GLOBAL_ARGS = None


def get_args():
    """Return arguments."""
    global _GLOBAL_ARGS
    _ensure_var_is_initialized(_GLOBAL_ARGS)
    return _GLOBAL_ARGS


def _ensure_var_is_initialized(var):
    """Make sure the input variable is not None."""
    assert var is not None, 'arguments are not initialized.'


def parse_args(extra_args_provider=None, defaults={},
               ignore_unknown_args=False, required_args={}, post_process_args=None):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Arguments',
                                     allow_abbrev=False)
    # Standard arguments.

    # parser = _add_LSTM_size_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_data_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_transformer_args(parser)
    parser = _add_lstm_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            print('WARNING: overriding default arguments for {key}:{v} \
                    with {key}:{v2}'.format(key=key, v=defaults[key],
                                            v2=getattr(args, key)),
                  flush=True)
        else:
            setattr(args, key, defaults[key])

    # Check required arguments.
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    if post_process_args is not None:
        args = post_process_args(args)

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    print(_GLOBAL_ARGS)
    return args


def print_args(args):
    """Print arguments."""
    log = get_logger()
    log.info('-------------------- arguments --------------------')
    str_list = []
    for arg in vars(args):
        dots = '.' * (32 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        log.info(str(arg))
    log.info('---------------- end of arguments ----------------')


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--test-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--optimizer-step-interval', type=int, default=1,
                       help='number of times of backwards for once step')
    group.add_argument('--device', type=int, default=0)
    group.add_argument('--getitem-method', type=str, default=None)
    group.add_argument('--collate-fn-method', type=str, default=None)
    group.add_argument('--train-epochs', type=int, default=None,
                       help='Total number of epochs to train over all '
                       'training runs.')
    group.add_argument('--go-on-train', type=int, default=0)
    group.add_argument('--fake', type=str, default="")
    group.add_argument('--take_output', type=str, default="maxpooling")
    group.add_argument('--activation', type=str, default="tanh")
    group.add_argument('--hidden', type=int, default=512)
    group.add_argument('--noglove', type=int, default=0)
    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=10000,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')
    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save-folder', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--no-save-optim', action='store_true',
                       help='Do not save current optimizer.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--train', type=int, default=1,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument("--lr-decay-epochs", type=tuple, default=(40, 80, 120, 160), help="where to decay lr, can be a list")
    group.add_argument("--lr-decay-rate", type=float, default=0.4, help="decay rate for learning rate")
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='Percentage of total iterations to warmup on '
                       '(.01 = 1 percent of all training iters).')
    group.add_argument('--override-lr-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-lr-scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')
    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    # group.add_argument('--attention-dropout', type=float, default=0.1,
    #                    help='Post attention dropout ptobability.')
    # group.add_argument('--hidden-dropout', type=float, default=0.1,
    #                    help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.00,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--embedding_dropout_prob', type=float, default=None,
                       help='embedding_dropout_prob.')

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument('--data-path', type=str, default="IMdata/",
                       help='Path to dataset.')
    group.add_argument('--num-workers', type=int, default=8,
                       help="Dataloader number of workers.")
    group.add_argument('--split-factor', type=float, default=0.85)
    group.add_argument('--Ngram', type=int, default=0)
    group.add_argument('--Punc', type=int, default=0)
    group.add_argument('--UpperCase', type=int, default=0)
    return parser


def _add_lstmtransformer_args(parser):
    group = parser.add_argument_group(
        title='arguments related to lstmtransformer')
    group.add_argument('--insulin-FE-dim', type=int, default=1)
    group.add_argument('--sugar-FE-dim', type=int, default=1)
    group.add_argument('--max-days', type=int, default=50)
    group.add_argument('--day-dim', type=int, default=50)
    group.add_argument('--time-dim', type=int, default=51)
    return parser


def _add_transformer_args(parser):
    group = parser.add_argument_group(
        title='some arguments related to transformer')
    group.add_argument('--tf-N', type=int, default=6)
    group.add_argument('--tf-d-hidden', type=int, default=100)
    group.add_argument('--tf-d-ff', type=int, default=100)
    group.add_argument('--tf-n-h', type=int, default=4)
    group.add_argument('--tf-dropout', type=float, default=0.35)
    return parser


def _add_lstm_args(parser):
    group = parser.add_argument_group(title='some arguments related to LSTM')
    group.add_argument('--lstm-N', type=int, default=6)
    group.add_argument('--lstm-d-hidden', type=int, default=6)
    group.add_argument('--examination-proj-dim', type=int, default=6)
    group.add_argument('--insulin-proj-dim', type=int, default=6)
    group.add_argument('--sugar-proj-dim', type=int, default=6)
    group.add_argument('--drug-proj-dim', type=int, default=6)
    return parser



