import sys
from torch.utils.data import DataLoader
import copy
import time
import os
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import argparse
import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from optax import adam
import optax
import typing as T

os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
from flax_mup import Mup, Readout

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
                import gc; gc.collect()
                variables = mup_state.set_target_shapes(variables)
            return model, variables, mup_state

        return f

    widths = 2 ** np.arange(7, 14)
    models = {w: gen(w, standparam=not mup) for w in widths}

    df = get_coord_data(models, train_loader,mup=mup, lr=lr, optimizer='adam', flatten_input=True, nseeds=nseeds,
                        nsteps=nsteps, lossfn='nll')

    prm = 'μP' if mup else 'SP'
    df = df.replace([np.inf,float('inf'),np.nan],1e99)
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
    parser.add_argument('--output_mult', type=float, default=1.0 #32
                        )
    parser.add_argument('--input_mult', type=float, default=1.0 #0.00390625
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
    parser.add_argument('--base_width',type=int,default=64)
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)

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

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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
            x= Readout(self.num_classes, use_bias=False)(x)  # 1. Replace output layer with Readout layer
            trace.append(x)
            return x, trace

        def scale_parameters(self, variables):
            variables['params']['Dense_0']['kernel'] /= self.input_mult ** 0.5
            variables['params']['Dense_0']['kernel'] *= self.init_std ** 0.5
            variables['params']['Dense_1']['kernel'] *= self.init_std ** 0.5
            # TODO: FC3 (readout) Zeros???
            return variables


    def train(args, model:MLP, device, train_loader, optimizer, epoch,
              scheduler=None, criterion=optax.softmax_cross_entropy):
        model.train()
        train_loss = 0
        correct = 0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.view(data.size(0), -1))

            loss = criterion(output, target)
            loss.backward()
            train_loss += loss.item() * data.shape[0]  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                elapsed = time.time() - start_time
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} | ms/batch {:5.2f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           elapsed * 1000 / args.log_interval))
                start_time = time.time()
            if scheduler is not None:
                scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
        return train_loss, train_acc


    def test(args, model, device, test_loader,
             evalmode=True, criterion=optax.softmax_cross_entropy):
        if evalmode:
            model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data.view(data.size(0), -1))
                test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss, correct / len(test_loader.dataset)


    def MSE_label(output, target):
        y_onehot = output.new_zeros(output.size(0), 10)
        y_onehot.scatter_(1, target.unsqueeze(-1), 1)
        y_onehot -= 1 / 10
        return F.mse_loss(output, y_onehot)


    if args.coord_check:
        print('testing parametrization')
        import os

        os.makedirs('coord_checks', exist_ok=True)
        plotdir = 'coord_checks'
        coord_check(mup=True, lr=args.lr, train_loader=train_loader, nsteps=args.coord_check_nsteps,
                    nseeds=args.coord_check_nseeds, args=args, plotdir=plotdir, legend=False)
        coord_check(mup=False, lr=args.lr, train_loader=train_loader, nsteps=args.coord_check_nsteps,
                    nseeds=args.coord_check_nseeds, args=args, plotdir=plotdir, legend=False)

        sys.exit()

    logs = []
    for nonlin in [torch.relu, torch.tanh]:
        for criterion in [F.cross_entropy, MSE_label]:

            for width in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                # print(f'{nonlin.__name__}_{criterion.__name__}_{str(width)}')
                if args.save_base_shapes:
                    print(f'saving base shapes at {args.save_base_shapes}')
                    if args.deferred_init:
                        from torchdistx.deferred_init import deferred_init

                        # We don't need to instantiate the base and delta models
                        # Note: this only works with torch nightly since unsqueeze isn't supported for fake tensors in stable
                        base_shapes = get_shapes(
                            deferred_init(MLP, width=width, nonlin=nonlin, output_mult=args.output_mult,
                                          input_mult=args.input_mult))
                        delta_shapes = get_shapes(
                            # just need to change whatever dimension(s) we are scaling
                            deferred_init(MLP, width=width + 1, nonlin=nonlin, output_mult=args.output_mult,
                                          input_mult=args.input_mult)
                        )
                    else:
                        base_shapes = get_shapes(
                            MLP(width=width, nonlin=nonlin, output_mult=args.output_mult, input_mult=args.input_mult))
                        delta_shapes = get_shapes(
                            # just need to change whatever dimension(s) we are scaling
                            MLP(width=width + 1, nonlin=nonlin, output_mult=args.output_mult,
                                input_mult=args.input_mult)
                        )
                    make_base_shapes(base_shapes, delta_shapes, savefile=args.save_base_shapes)
                    print('done and exit')
                    import sys;

                    sys.exit()
                mynet = MLP(width=width, nonlin=nonlin, output_mult=args.output_mult, input_mult=args.input_mult).to(
                    device)
                if args.load_base_shapes:
                    print(f'loading base shapes from {args.load_base_shapes}')
                    set_base_shapes(mynet, args.load_base_shapes)
                    print('done')
                else:
                    print(f'using own shapes')
                    set_base_shapes(mynet, None)
                    print('done')
                optimizer = MuSGD(mynet.parameters(), lr=args.lr, momentum=args.momentum)
                for epoch in range(1, args.epochs + 1):
                    train_loss, train_acc, = train(args, mynet, device, train_loader, optimizer, epoch,
                                                   criterion=criterion)
                    test_loss, test_acc = test(args, mynet, device, test_loader)
                    logs.append(dict(
                        epoch=epoch,
                        train_loss=train_loss,
                        train_acc=train_acc,
                        test_loss=test_loss,
                        test_acc=test_acc,
                        width=width,
                        nonlin=nonlin.__name__,
                        criterion='xent' if criterion.__name__ == 'cross_entropy' else 'mse',
                    ))
                    if math.isnan(train_loss):
                        break

    with open(os.path.join(os.path.expanduser(args.log_dir), 'logs.tsv'), 'w') as f:
        logdf = pd.DataFrame(logs)
        print(os.path.join(os.path.expanduser(args.log_dir), 'logs.tsv'))
        f.write(logdf.to_csv(sep='\t', float_format='%.4f'))
