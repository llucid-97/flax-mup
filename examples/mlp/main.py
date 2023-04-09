from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets, transforms
import argparse
import jax
import jax.numpy as jnp
from flax import linen as nn
import typing as T

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
from flax_mup.coord_check import get_coord_data, plot_coord_data
from flax_mup import Mup, Readout


def coord_check(mup: bool, lr, train_loader, nsteps, nseeds, args, plotdir='', legend=False):
    def gen(w, standparam=False):
        def f():
            model = MLP(width=w,
                        nonlin=nn.tanh,
                        output_mult=args.output_mult,
                        input_mult=args.input_mult)

            init_input = jnp.zeros((1, 3072))
            variables = model.init(jax.random.PRNGKey(0), init_input)
            variables = model.scale_parameters(variables.unfreeze())
            if standparam:
                mup_state = None
            else:
                mup_state = Mup()
                base_model = MLP(width=args.base_width,
                                 nonlin=nn.tanh,
                                 output_mult=args.output_mult,
                                 input_mult=args.input_mult)
                base_vars = base_model.init(jax.random.PRNGKey(0), init_input)
                base_vars = model.scale_parameters(base_vars.unfreeze())

                mup_state.set_base_shapes(base_vars)
                del base_model, base_vars
                import gc;
                gc.collect()
                variables = mup_state.set_target_shapes(variables)
            return model, variables, mup_state

        return f

    widths = 2 ** np.arange(7, 14)
    models = {w: gen(w, standparam=not mup) for w in widths}

    df = get_coord_data(models, train_loader, mup=mup, lr=lr, optimizer='adam', flatten_input=True, nseeds=nseeds,
                        nsteps=nsteps, lossfn='nll')

    prm = 'μP' if mup else 'SP'
    df = df.replace([np.inf, float('inf'), np.nan], 1e99)
    return plot_coord_data(df, legend=legend,
                           save_to=os.path.join(plotdir, f'{prm.lower()}_mlp_sgd_coord.png'),
                           suptitle=f'{prm} MLP SGD lr={lr} nseeds={nseeds}',
                           face_color='xkcd:light grey' if not mup else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    PyTorch MLP on CIFAR-10, with μP.

    This is the script we use in the MLP experiment in our paper.

    To train a μP model, one needs to first specify the base shapes. To save base shapes info, run, for example,

        python main.py --save_base_shapes width64.bsh

    To train using MuSGD, run

        python main.py --load_base_shapes width64.bsh

    To perform coord check, run

        python main.py --load_base_shapes width64.bsh --coord_check

    If you don't specify a base shape file, then you are using standard parametrization

        python main.py

    We provide below some optimal hyperparameters for different activation/loss function combos:
        if nonlin == torch.relu and criterion == F.cross_entropy:
            args.input_mult = 0.00390625
            args.output_mult = 32
        elif nonlin == torch.tanh and criterion == F.cross_entropy:
            args.input_mult = 0.125
            args.output_mult = 32
        elif nonlin == torch.relu and criterion == MSE_label:
            args.input_mult = 0.03125
            args.output_mult = 32
        elif nonlin == torch.tanh and criterion == MSE_label:
            args.input_mult = 8
            args.output_mult = 0.125
    ''', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--output_mult', type=float, default=1.0  # 32
                        )
    parser.add_argument('--input_mult', type=float, default=1.0  # 0.00390625
                        )
    parser.add_argument('--init_std', type=float, default=1.0)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--log_interval', type=int, default=300)
    parser.add_argument('--log_dir', type=str, default='.')
    parser.add_argument('--data_dir', type=str, default='/tmp')
    parser.add_argument('--coord_check', action='store_true',
                        help='test μ parametrization is correctly implemented by collecting statistics on coordinate distributions for a few steps of training.')
    parser.add_argument('--coord_check_nsteps', type=int, default=3,
                        help='Do coord check with this many steps.')
    parser.add_argument('--coord_check_nseeds', type=int, default=5,
                        help='number of seeds for testing correctness of μ parametrization')
    parser.add_argument('--deferred_init', action='store_true',
                        help='Skip instantiating the base and delta models for mup. Requires torchdistx.')
    parser.add_argument('--base_width', type=int, default=64)
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=args.data_dir, train=True,
                                download=True, transform=transform)
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=not args.no_shuffle,
                              num_workers=2)

    testset = datasets.CIFAR10(root=args.data_dir, train=False,
                               download=True, transform=transform)
    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=2)


    class MLP(nn.Module):
        width: int = 128
        num_classes: int = 10
        nonlin: T.Callable = nn.relu
        output_mult: float = 1.0
        input_mult: float = 1.0
        init_std: float = 1.0

        @nn.compact
        def __call__(self, x):
            trace = [x]
            x = nn.Dense(self.width, use_bias=False, kernel_init=nn.initializers.kaiming_normal())(x)
            trace.append(x)
            x = x * self.input_mult ** 0.5
            x = self.nonlin(x)
            trace.append(x)
            x = nn.Dense(self.width, use_bias=False, kernel_init=nn.initializers.kaiming_normal())(x)
            trace.append(x)
            x = self.nonlin(x)
            x = x * self.output_mult
            trace.append(x)
            x = Readout(self.num_classes, use_bias=False)(x)  # 1. Replace output layer with Readout layer
            trace.append(x)
            return x, trace

        def scale_parameters(self, variables):
            variables['params']['Dense_0']['kernel'] /= self.input_mult ** 0.5
            variables['params']['Dense_0']['kernel'] *= self.init_std ** 0.5
            variables['params']['Dense_1']['kernel'] *= self.init_std ** 0.5
            return variables


    print('testing parametrization')
    import os

    os.makedirs('coord_checks', exist_ok=True)
    plotdir = 'coord_checks'
    coord_check(mup=True, lr=args.lr, train_loader=train_loader, nsteps=args.coord_check_nsteps,
                nseeds=args.coord_check_nseeds, args=args, plotdir=plotdir, legend=False)
    coord_check(mup=False, lr=args.lr, train_loader=train_loader, nsteps=args.coord_check_nsteps,
                nseeds=args.coord_check_nseeds, args=args, plotdir=plotdir, legend=False)
