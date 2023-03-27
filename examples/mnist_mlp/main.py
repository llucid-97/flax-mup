import time
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
from torch import nn
import torch.optim as optim
import argparse
import math

from mup.coord_check import get_coord_data, plot_coord_data
from mup import MuSGD, get_shapes, set_base_shapes, make_base_shapes, MuReadout


def coord_check(mup, lr, train_loader, nsteps, nseeds, args, plotdir='', legend=False):
    def gen(w, standparam=False):
        def f():
            model = MLP(width=w, nonlin=torch.tanh, output_mult=args.output_mult, input_mult=args.input_mult).to(device)
            if standparam:
                set_base_shapes(model, None)
            else:
                assert args.load_base_shapes, 'load_base_shapes needs to be nonempty'
                set_base_shapes(model, args.load_base_shapes)
            return model

        return f

    widths = 2 ** np.arange(7, 14)
    models = {w: gen(w, standparam=not mup) for w in widths}

    df = get_coord_data(models, train_loader, mup=mup, lr=lr, optimizer='sgd', flatten_input=True, nseeds=nseeds,
                        nsteps=nsteps, lossfn='nll')

    prm = 'μP' if mup else 'SP'
    return plot_coord_data(df, legend=legend,
                           save_to=os.path.join(plotdir, f'{prm.lower()}_mlp_sgd_coord.png'),
                           suptitle=f'{prm} MLP SGD lr={lr} nseeds={nseeds}',
                           face_color='xkcd:light grey' if not mup else None)


if __name__ == '__main__':
    def get_args():
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
        parser.add_argument('--save_base_shapes', type=str, default='',
                            help='file location to save base shapes at')
        parser.add_argument('--load_base_shapes', type=str, default='',
                            help='file location to load base shapes from')
        parser.add_argument('--seed', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--output_mult', type=float, default=1.0)
        parser.add_argument('--input_mult', type=float, default=1.0)
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

        args = parser.parse_args()
        return args

    args = get_args()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=args.data_dir, train=True,
                                download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=not args.no_shuffle, num_workers=2)

    testset = datasets.CIFAR10(root=args.data_dir, train=False,
                               download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    class MLP(nn.Module):
        def __init__(self, width=128, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0):
            super(MLP, self).__init__()
            self.nonlin = nonlin
            self.input_mult = input_mult
            self.output_mult = output_mult
            self.fc_1 = nn.Linear(3072, width, bias=False)
            self.fc_2 = nn.Linear(width, width, bias=False)
            self.fc_3 = MuReadout(width, num_classes, bias=False, output_mult=args.output_mult)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
            self.fc_1.weight.data /= self.input_mult ** 0.5
            self.fc_1.weight.data *= args.init_std
            nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
            self.fc_2.weight.data *= args.init_std
            nn.init.zeros_(self.fc_3.weight)

        def forward(self, x):
            out = self.nonlin(self.fc_1(x) * self.input_mult ** 0.5)
            out = self.nonlin(self.fc_2(out))
            return self.fc_3(out)


    def train(args, model, device, train_loader, optimizer, epoch,
              scheduler=None, criterion=F.cross_entropy):
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
             evalmode=True, criterion=F.cross_entropy):
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
        import sys;

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

from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                        config.batch_size,
                                                        input_rng)
        _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
                                                  test_ds['label'])

        logging.info(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
            % (epoch, train_loss, train_accuracy * 100, test_loss,
               test_accuracy * 100))

        summary_writer.scalar('train_loss', train_loss, epoch)
        summary_writer.scalar('train_accuracy', train_accuracy, epoch)
        summary_writer.scalar('test_loss', test_loss, epoch)
        summary_writer.scalar('test_accuracy', test_accuracy, epoch)

    summary_writer.flush()
    return state
