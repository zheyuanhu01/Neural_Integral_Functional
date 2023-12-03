import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp
from typing import Any, Sequence, Union
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import argparse

class MLP(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x, n, nabla_n):
        x = jnp.hstack([x, n, nabla_n])
        for dim in self.layers[:-1]:
            x = hk.Linear(dim)(x)
            x = jax.nn.gelu(x)
        x = hk.Linear(self.layers[-1])(x)
        return x[0]

class PolyNet(hk.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def __call__(self, x):
        k = np.arange(self.N)
        h = jnp.hstack([x**k])
        h = hk.Linear(1, with_bias=False)(h)
        x = h * (1 - x**2)
        print(x.shape)
        return x

class SpectralConv1d(hk.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = (1 / (in_channels))

    def __call__(self, x):
        batchsize = jnp.shape(x)[0]
        x_ft = jnp.fft.rfft(x)
        w1_init = hk.initializers.RandomUniform(maxval=self.scale) #ComplexRandomUniform(maxval=self.scale)
        w2_init = hk.initializers.Constant(constant=0)
        weights1 = hk.get_parameter("weights1", shape=[self.in_channels, self.out_channels, self.modes1], dtype=x.dtype, init=w1_init)
        weights2 = hk.get_parameter("weights2", shape=[self.in_channels, self.out_channels, self.modes1], dtype=x.dtype, init=w2_init)
        out_ft = jnp.zeros((batchsize, self.out_channels, jnp.shape(x)[-1] // 2 + 1), dtype=x_ft.dtype)
        out_ft = out_ft.at[:, :, :self.modes1].set(jnp.einsum("bix,iox->box", x_ft[:, :, :self.modes1], weights1))
        out_ft = out_ft.at[:, :, :self.modes1].add(jnp.einsum("bix,iox->box", x_ft[:, :, :self.modes1], weights2 * 1.j))
        x = jnp.fft.irfft(out_ft, n=jnp.shape(x)[-1])
        return x

class FNO1d(hk.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = hk.Linear(self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = hk.Conv1D(self.width, 1, data_format='NCW')
        self.w1 = hk.Conv1D(self.width, 1, data_format='NCW')
        self.w2 = hk.Conv1D(self.width, 1, data_format='NCW')
        self.w3 = hk.Conv1D(self.width, 1, data_format='NCW')

        self.fc1 = hk.Linear(128)
        self.fc2 = hk.Linear(1)

    def __call__(self, x, n, nabla_n):
        x = jnp.concatenate([x, n], axis=-1)
        x = jnp.expand_dims(x, 0)
        x = self.fc0(x)
        x = jnp.transpose(x, (0, 2, 1))

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = jax.nn.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = jax.nn.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = jax.nn.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = jnp.transpose(x, (0, 2, 1))
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        x = jnp.reshape(x, (-1))
        return x

def get_grid(gridx, shape):
    batchsize, size_x = shape[0], shape[1]
    #gridx = jnp.linspace(0, 1, size_x)
    gridx = jnp.reshape(gridx, (1, size_x, 1))
    gridx = jnp.repeat(gridx, batchsize, axis=0)
    #gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
    return gridx

def kinetic_energy(x, w, B=10000, N=20, ve=False):
    a = (np.random.rand(B, N) - 0.5) * 2
    def poly(x):
        return (a @ (np.reshape(x, (-1, 1)) ** np.arange(N)).T)
    def nabla_poly(x):
        return (a[:, 1:] @ (np.arange(1, N).reshape(-1, 1) * (np.reshape(x, (-1, 1)) ** np.arange(N-1)).T))
    def nabla2_poly(x):
        return (a[:, 2:] @ (np.arange(2, N).reshape(-1, 1) * np.arange(1, N-1).reshape(-1, 1) * \
            (np.reshape(x, (-1, 1)) ** np.arange(N-2)).T))
    n, nabla_n, nabla2_n = poly(x), nabla_poly(x), nabla2_poly(x)
    y = np.sum(0.5 * w.reshape(1, -1) * nabla_poly(x)**2, axis=-1)
    n, nabla_n, nabla2_n = np.expand_dims(n, axis=-1), np.expand_dims(nabla_n, axis=-1), np.expand_dims(nabla2_n, axis=-1)
    m = 0.5 * nabla_n**2
    dy = -nabla2_n
    train_n, train_nabla_n, train_y, test_n, test_nabla_n, test_y = n[:int(0.9 * B), :, :], nabla_n[:int(0.9 * B), :, :], \
        y[:int(0.9 * B)], n[int(0.9 * B):, :, :], nabla_n[int(0.9 * B):, :, :], y[int(0.9 * B):]
    train_dy, test_dy = dy[:int(0.9 * B)], dy[int(0.9 * B):]
    train_nabla2_n, test_nabla2_n = nabla2_n[:int(0.9 * B)], nabla2_n[int(0.9 * B):]
    train_m, test_m = m[:int(0.9 * B)], m[int(0.9 * B):]
    return train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy, test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy

def shortest_path(x, w, B=10000, N=20, ve=False):
    a = (np.random.rand(B, N) - 0.5) * 2
    def poly(x):
        return (a @ (np.reshape(x, (-1, 1)) ** np.arange(N)).T)
    def nabla_poly(x):
        return (a[:, 1:] @ (np.arange(1, N).reshape(-1, 1) * (np.reshape(x, (-1, 1)) ** np.arange(N-1)).T))
    def nabla2_poly(x):
        return (a[:, 2:] @ (np.arange(2, N).reshape(-1, 1) * np.arange(1, N-1).reshape(-1, 1) * \
            (np.reshape(x, (-1, 1)) ** np.arange(N-2)).T))
    n, nabla_n, nabla2_n = poly(x), nabla_poly(x), nabla2_poly(x)
    y = np.sum(w.reshape(1, -1) * np.sqrt(1 + nabla_poly(x)**2), axis=-1)
    n, nabla_n, nabla2_n = np.expand_dims(n, axis=-1), np.expand_dims(nabla_n, axis=-1), np.expand_dims(nabla2_n, axis=-1)
    m = np.sqrt(1 + nabla_n**2)
    dy = - nabla2_n / (1 + nabla_n**2)**(1.5)
    train_n, train_nabla_n, train_y, test_n, test_nabla_n, test_y = n[:int(0.9 * B), :, :], nabla_n[:int(0.9 * B), :, :], \
        y[:int(0.9 * B)], n[int(0.9 * B):, :, :], nabla_n[int(0.9 * B):, :, :], y[int(0.9 * B):]
    train_dy, test_dy = dy[:int(0.9 * B)], dy[int(0.9 * B):]
    train_nabla2_n, test_nabla2_n = nabla2_n[:int(0.9 * B)], nabla2_n[int(0.9 * B):]
    train_m, test_m = m[:int(0.9 * B)], m[int(0.9 * B):]
    return train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy, test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy

def poisson(x, w, B=10000, N=20, ve=False, type_eq=0):
    a = (np.random.rand(B, N) - 0.5) * 2
    def poly(x):
        return (a @ (np.reshape(x, (-1, 1)) ** np.arange(N)).T)
    def nabla_poly(x):
        return (a[:, 1:] @ (np.arange(1, N).reshape(-1, 1) * (np.reshape(x, (-1, 1)) ** np.arange(N-1)).T))
    def nabla2_poly(x):
        return (a[:, 2:] @ (np.arange(2, N).reshape(-1, 1) * np.arange(1, N-1).reshape(-1, 1) * \
            (np.reshape(x, (-1, 1)) ** np.arange(N-2)).T))
    n = poly(x) * (1 - x**2).reshape(1, -1)
    nabla_n = nabla_poly(x) * (1 - x**2).reshape(1, -1) - poly(x) * (2 * x).reshape(1, -1)
    nabla2_n = nabla2_poly(x) * (1 - x**2).reshape(1, -1) - nabla_poly(x) * (2 * x).reshape(1, -1) \
         - 2 * poly(x) - nabla_poly(x) * (2 * x).reshape(1, -1)
    if type_eq == 0:
        f = (12 * x**2 - 4).reshape(1, -1)
    elif type_eq == 1:
        # (0.8*x**5 + 0.9*x**4)*(x**2-1)
        f = (7*6*0.8*x**5 + 0.9*6*5*x**4 - 0.8*5*4*x**3 - 0.9*4*3*x**2)
    else:
        #f = (-0.8*x**7 + x**2)*(x**2-1)
        f = (-9*8*0.8*x**7 + 4*3*x**2 + 0.8*7*6*x**5 - 2)
    y = np.sum(w.reshape(1, -1) * (0.5 * nabla_n**2 - f * n), axis=-1)
    m = 0.5 * nabla_n**2 - f * n
    n, nabla_n, nabla2_n = np.expand_dims(n, axis=-1), np.expand_dims(nabla_n, axis=-1), np.expand_dims(nabla2_n, axis=-1)
    dy = - nabla2_n - f.reshape(1, -1, 1)
    train_n, train_nabla_n, train_y, test_n, test_nabla_n, test_y = n[:int(0.9 * B), :, :], nabla_n[:int(0.9 * B), :, :], \
        y[:int(0.9 * B)], n[int(0.9 * B):, :, :], nabla_n[int(0.9 * B):, :, :], y[int(0.9 * B):]
    train_dy, test_dy = dy[:int(0.9 * B)], dy[int(0.9 * B):]
    train_nabla2_n, test_nabla2_n = nabla2_n[:int(0.9 * B)], nabla2_n[int(0.9 * B):]
    train_m, test_m = m[:int(0.9 * B)], m[int(0.9 * B):]
    return train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy, test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Functional Project')
    parser.add_argument('--N_data', type=int, default=1100)
    parser.add_argument('--N_grid', type=int, default=256)
    parser.add_argument('--dataset', type=str, default="poisson") # sp
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100001)
    parser.add_argument('--lam_f', type=float, default=0)
    parser.add_argument('--ve', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--type_eq', type=int, default=0)
    args = parser.parse_args()
    key = jax.random.PRNGKey(args.seed)
    print(args.ve)
    if args.dataset == "ke":
        x, w = np.polynomial.legendre.leggauss(args.N_grid)
        x, w = (x + 1) / 2, w / 2
        train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy, test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy = kinetic_energy(x, w, B=args.N_data, ve=args.ve)
    elif args.dataset == "sp":
        x, w = np.polynomial.legendre.leggauss(args.N_grid)
        x, w = (x + 1) / 2, w / 2
        train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy, test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy = shortest_path(x, w, B=args.N_data, ve=args.ve)
    elif args.dataset == "poisson":
        x, w = np.polynomial.legendre.leggauss(args.N_grid)
        train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy, test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy = poisson(x, w, B=args.N_data, ve=args.ve, type_eq=args.type_eq)
    gridx = x
    print(train_n.shape, train_nabla_n.shape, train_nabla2_n.shape, train_m.shape, train_y.shape, train_dy.shape, test_n.shape, test_nabla_n.shape, test_nabla2_n.shape, test_m.shape, test_y.shape, test_dy.shape)
    @hk.transform
    def network(x, n, nabla_n):
        temp = FNO1d(8, 32) 
        return temp(x, n, nabla_n)
    net = hk.without_apply_rng(network)

    """train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy, test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy = \
        train_n.reshape(-1), train_nabla_n.reshape(-1), train_nabla2_n.reshape(-1), train_m.reshape(-1), train_y.reshape(-1), train_dy.reshape(-1), test_n.reshape(-1), test_nabla_n.reshape(-1), test_nabla2_n.reshape(-1), test_m.reshape(-1), test_y.reshape(-1), test_dy.reshape(-1)
        #train_n.reshape(-1, 1), train_nabla_n.reshape(-1, 1), train_nabla2_n.reshape(-1, 1), train_m.reshape(-1, 1), train_y.reshape(-1, 1), train_dy.reshape(-1, 1), test_n.reshape(-1, 1), test_nabla_n.reshape(-1, 1), test_nabla2_n.reshape(-1, 1), test_m.reshape(-1, 1), test_y.reshape(-1, 1), test_dy.reshape(-1, 1)"""

    x = get_grid(gridx, jnp.shape(train_n))#get_grid(gridx, jnp.shape(train_n)[0] // args.N_grid, args.N_grid)
    params = net.init(key, x[0], train_n[0], train_nabla_n[0])
    net_pred_fn = jax.vmap(net.apply, (None, 0, 0, 0))

    optimizer = optax.adam(1e-3)
    lr = optax.exponential_decay(init_value=1e-3, transition_steps=5000, decay_rate=0.9)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def get_loss(params, x, n, nabla_n, nabla2_n, m, dy):
        y_pred = net_pred_fn(params, x, n, nabla_n)
        ret = jnp.mean((y_pred - m)**2) 
        return ret

    @jax.jit
    def test_loss(params, x, n, nabla_n, nabla2_n, m, y, dy):
        y_pred = net_pred_fn(params, x, n, nabla_n)
        i_pred = jnp.sum(jnp.reshape(w.reshape(1, -1) * jnp.reshape(y_pred, (-1, args.N_grid)), (-1, args.N_grid)), axis=1)
        return y_pred, i_pred

    @jax.jit
    def step(params, opt_state, x, n, nabla_n, nabla2_n, m, dy):
        current_loss, gradients = jax.value_and_grad(get_loss)(params, x, n, nabla_n, nabla2_n, m, dy)
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state
    
    for epoch in tqdm(range(args.epochs)):
        x = get_grid(gridx, jnp.shape(train_n))
        current_loss, params, opt_state = step(params, opt_state, x, train_n, train_nabla_n, train_nabla2_n, train_m, train_dy)
        if epoch%1000==0:
            x = get_grid(gridx, jnp.shape(test_n))
            pred, pred_i = test_loss(params, x, test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy)
            err_func = jnp.linalg.norm(jnp.reshape(pred - test_m, (-1,))) / jnp.linalg.norm(jnp.reshape(test_m, (-1,)))
            err_intor = jnp.linalg.norm(jnp.reshape(pred_i - test_y, (-1,))) / jnp.linalg.norm(jnp.reshape(test_y, (-1,)))
            print('epoch %d, loss: %.3E, err_func: %.3E, err_intor: %.3E'%(epoch, current_loss, err_func, err_intor))

    if args.dataset == "poisson":
        @hk.transform
        def poly_network(x):
            temp = PolyNet(N=20)
            return temp(x)
        poly_net = hk.without_apply_rng(poly_network)
        x = get_grid(gridx, jnp.shape(train_n))
        poly_params = poly_net.init(key, x[0])
        poly_net_pred_fn = jax.vmap(poly_net.apply, (None, 0))
        nabla_poly_net = jax.vmap(jax.grad(poly_net.apply, argnums=1), (None, 0))
        nabla2_poly_net = jax.vmap(jax.hessian(poly_net.apply, argnums=1), (None, 0))

        lr = optax.exponential_decay(init_value=1e-3, transition_steps=5000, decay_rate=0.9)
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(poly_params)

        @jax.jit
        def get_loss_poly(poly_params, x):
            y_pred = poly_net_pred_fn(poly_params, x)
            dy_pred = y_pred# nabla_poly_net(poly_params, x)
            #d2y_pred = nabla2_poly_net(poly_params, x)
            loss = net_pred_fn(params, x, y_pred, dy_pred)
            loss = w.reshape(1, -1) * jnp.reshape(loss, (-1, args.N_grid))
            loss = jnp.sum(loss, axis=-1)
            loss = jnp.mean(loss)
            return loss
        
        @jax.jit
        def step_poly(poly_params, opt_state, x):
            current_loss, gradients = jax.value_and_grad(get_loss_poly)(poly_params, x)
            updates, opt_state = optimizer.update(gradients, opt_state)
            poly_params = optax.apply_updates(poly_params, updates)
            return current_loss, poly_params, opt_state
        
        for epoch in tqdm(range(args.epochs)):
            x = get_grid(gridx, jnp.shape(train_n))
            current_loss, poly_params, opt_state = step_poly(poly_params, opt_state, x)
            if epoch%1000==0:
                show_params = poly_params['poly_net/linear']['w']
                show_params = jnp.reshape(show_params, (-1))
                print(show_params)
                print(current_loss)
        
        x = get_grid(gridx, jnp.shape(train_n))
        y_pred = poly_net_pred_fn(poly_params, x)
        if args.type_eq == 0:
            y_true = x**4 - 2 * x**2 + 1
        elif args.type_eq == 1:
            y_true = (0.8*x**5 + 0.9*x**4)*(x**2-1)
        else:
            y_true = (-0.8*x**7 + x**2)*(x**2-1)
        print(y_true.shape, y_pred.shape)
        print("relative L2 error: ", jnp.linalg.norm(y_pred + y_true) / jnp.linalg.norm(y_true))
        #np.savetxt("x.txt", np.asarray(x)[:args.N_grid])
        np.savetxt("y_true_"+str(args.type_eq)+"_FNO.txt", np.asarray(y_true)[0, :args.N_grid])
        np.savetxt("y_pred_"+str(args.type_eq)+"_FNO.txt", np.asarray(y_pred)[0, :args.N_grid])
        pass
    