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

def get_grid(batchsize, size_x):
    gridx = jnp.linspace(0, 1, size_x)
    gridx = jnp.reshape(gridx, (1, size_x, 1))
    gridx = jnp.repeat(gridx, batchsize, axis=0)
    gridx = jnp.reshape(gridx, (batchsize * size_x,))
    #gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
    return gridx

def polynomial(B=10000, N=20, N_grid=1000):
    a = (np.random.rand(B, N) - 0.5) * 2
    def poly(x):
        return (a @ (np.reshape(x, (-1, 1)) ** np.arange(N)).T)
    def nabla_poly(x):
        return (a[:, 1:] @ (np.arange(1, N).reshape(-1, 1) * (np.reshape(x, (-1, 1)) ** np.arange(N-1)).T))
    def nabla2_poly(x):
        return (a[:, 2:] @ (np.arange(2, N).reshape(-1, 1) * np.arange(1, N-1).reshape(-1, 1) * \
            (np.reshape(x, (-1, 1)) ** np.arange(N-2)).T))
    def integrate():
        return (a @ np.reshape(1. / np.arange(1, N+1), (-1, 1)))
    x = np.linspace(0, 1, N_grid)
    n, nabla_n, nabla2_n = poly(x), nabla_poly(x), nabla2_poly(x)
    n, nabla_n, nabla2_n = np.expand_dims(n, axis=-1), np.expand_dims(nabla_n, axis=-1), np.expand_dims(nabla2_n, axis=-1)
    m = n
    y = integrate()
    #print(np.mean(y), np.std(y))
    """y = poly(x)
    i1 = np.mean(y, axis=-1)
    print(i1, integrate())"""
    dy = np.ones_like(n)
    """train_n, train_nabla_n, train_y, test_n, test_nabla_n, test_y = n[:int(0.9 * B), :, :], nabla_n[:int(0.9 * B), :, :], \
        y[:int(0.9 * B)], n[int(0.9 * B):, :, :], nabla_n[int(0.9 * B):, :, :], y[int(0.9 * B):]
    train_dy, test_dy = dy[:int(0.9 * B)], dy[int(0.9 * B):]
    train_nabla2_n, test_nabla2_n = nabla2_n[:int(0.9 * B)], nabla2_n[int(0.9 * B):]
    train_m, test_m = m[:int(0.9 * B)], m[int(0.9 * B):]"""
    return n, nabla_n, nabla2_n, m, y, dy

def polynomial1(B=10000, N=20, N_grid=1000):
    a = (np.random.rand(B, N) - 0.5) * 2
    def poly(x):
        return (a @ (np.reshape(x, (-1, 1)) ** np.arange(N)).T)
    def nabla_poly(x):
        return (a[:, 1:] @ (np.arange(1, N).reshape(-1, 1) * (np.reshape(x, (-1, 1)) ** np.arange(N-1)).T))
    def nabla2_poly(x):
        return (a[:, 2:] @ (np.arange(2, N).reshape(-1, 1) * np.arange(1, N-1).reshape(-1, 1) * \
            (np.reshape(x, (-1, 1)) ** np.arange(N-2)).T))
    def integrand(x):
        return (a @ (np.arange(N).reshape(-1, 1) * (np.reshape(x, (-1, 1)) ** np.arange(N)).T))
    def integrate():
        return (a[:, 1:] @ np.reshape(np.arange(1, N) / (np.arange(2, N+1) + 0.0), (-1, 1)))
    x = np.linspace(0, 1, N_grid)
    n, nabla_n, nabla2_n = poly(x), nabla_poly(x), nabla2_poly(x)
    n, nabla_n, nabla2_n = np.expand_dims(n, axis=-1), np.expand_dims(nabla_n, axis=-1), np.expand_dims(nabla2_n, axis=-1)
    y = integrate()
    m = x.reshape(1, -1, 1) * nabla_n
    """print(np.mean(y), np.std(y))
    y = integrand(np.linspace(0, 1, 10000))
    i1 = np.mean(y, axis=-1)
    print(i1, integrate())"""
    dy = -np.ones_like(n)
    """train_n, train_nabla_n, train_y, test_n, test_nabla_n, test_y = n[:int(0.9 * B), :, :], nabla_n[:int(0.9 * B), :, :], \
        y[:int(0.9 * B)], n[int(0.9 * B):, :, :], nabla_n[int(0.9 * B):, :, :], y[int(0.9 * B):]
    train_dy, test_dy = dy[:int(0.9 * B)], dy[int(0.9 * B):]
    train_nabla2_n, test_nabla2_n = nabla2_n[:int(0.9 * B)], nabla2_n[int(0.9 * B):]
    train_m, test_m = m[:int(0.9 * B)], m[int(0.9 * B):]"""
    return n, nabla_n, nabla2_n, m, y, dy

def RFF(B=10000, N=20, N_grid=1000):
    a = (np.random.rand(B, N, 1) - 0.5) * 2
    b = np.random.rand(B, N, 1) * 2 + 0.1
    def sinosoid(x):
        return np.sum(a * np.cos(b @ np.reshape(x, (1, -1))), axis=1)
    def nabla_sin(x):
        return - np.sum(b * a * np.sin(b @ np.reshape(x, (1, -1))), axis=1)
    def nabla2_sin(x):
        return - np.sum(b * b * a * np.cos(b @ np.reshape(x, (1, -1))), axis=1)
    def integrate(x):
        return np.sum(a / b * np.sin(b @ np.reshape(x, (1, -1))), axis=1)
    x = np.linspace(0, 1, N_grid)
    n, nabla_n, nabla2_n = sinosoid(x), nabla_sin(x), nabla2_sin(x)
    #print(n.mean(-1))
    n, nabla_n, nabla2_n = np.expand_dims(n, axis=-1), np.expand_dims(nabla_n, axis=-1), np.expand_dims(nabla2_n, axis=-1)
    y = integrate(np.ones(1)) - integrate(np.zeros(1))
    m = n
    #print(y)
    print(np.mean(y), np.std(y), np.max(y))
    #y = poly(x)
    #i1 = jnp.mean(y, axis=-1)
    #print(i1, integrate())
    dy = np.ones_like(n)
    """train_n, train_nabla_n, train_y, test_n, test_nabla_n, test_y = n[:int(0.9 * B), :, :], nabla_n[:int(0.9 * B), :, :], \
        y[:int(0.9 * B)], n[int(0.9 * B):, :, :], nabla_n[int(0.9 * B):, :, :], y[int(0.9 * B):]
    train_dy, test_dy = dy[:int(0.9 * B)], dy[int(0.9 * B):]
    train_nabla2_n, test_nabla2_n = nabla2_n[:int(0.9 * B)], nabla2_n[int(0.9 * B):]
    train_m, test_m = m[:int(0.9 * B)], m[int(0.9 * B):]"""
    return n, nabla_n, nabla2_n, m, y, dy

def RFF1(B=10000, N=20, N_grid=1000):
    a = (np.random.rand(B, N, 1) - 0.5) * 2
    b = np.random.rand(B, N, 1) * 2 + 0.1
    def sinosoid(x):
        return np.sum(a * np.cos(b @ np.reshape(x, (1, -1))), axis=1)
    def nabla_sin(x):
        return - np.sum(b * a * np.sin(b @ np.reshape(x, (1, -1))), axis=1)
    def nabla2_sin(x):
        return - np.sum(b * b * a * np.cos(b @ np.reshape(x, (1, -1))), axis=1)
    def integrate(x):
        return np.sum(a / b * np.sin(b @ np.reshape(x, (1, -1))), axis=1)
    def integrand(x):
        x = np.reshape(x, (1, -1))
        return - np.sum(b * a * np.sin(b @ x) * x, axis=1)
    x = np.linspace(0, 1, N_grid)
    n, nabla_n, nabla2_n = sinosoid(x), nabla_sin(x), nabla2_sin(x)
    #print(integrand(x).mean(-1))
    n, nabla_n, nabla2_n = np.expand_dims(n, axis=-1), np.expand_dims(nabla_n, axis=-1), np.expand_dims(nabla2_n, axis=-1)
    y = sinosoid(np.ones(1)) - integrate(np.ones(1)) + integrate(np.zeros(1))
    #print(y)
    m = x.reshape(1, -1, 1) * nabla_n
    #print(x.shape, nabla_n.shape, m.shape)
    print(np.mean(y), np.std(y), np.max(y))
    #y = poly(x)
    #i1 = jnp.mean(y, axis=-1)
    #print(i1, integrate())
    dy = -np.ones_like(n)
    """train_n, train_nabla_n, train_y, test_n, test_nabla_n, test_y = n[:int(0.9 * B), :, :], nabla_n[:int(0.9 * B), :, :], \
        y[:int(0.9 * B)], n[int(0.9 * B):, :, :], nabla_n[int(0.9 * B):, :, :], y[int(0.9 * B):]
    train_dy, test_dy = dy[:int(0.9 * B)], dy[int(0.9 * B):]
    train_nabla2_n, test_nabla2_n = nabla2_n[:int(0.9 * B)], nabla2_n[int(0.9 * B):]
    train_m, test_m = m[:int(0.9 * B)], m[int(0.9 * B):]"""
    return n, nabla_n, nabla2_n, m, y, dy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Functional Project')
    parser.add_argument('--N_data', type=int, default=1100)
    parser.add_argument('--N_grid', type=int, default=256)
    parser.add_argument('--dataset', type=str, default="poly")
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=20001)
    parser.add_argument('--lam_f', type=float, default=0)
    parser.add_argument('--order', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    key = jax.random.PRNGKey(args.seed)
    if args.order == 0 and args.dataset == "poly":
        train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy = polynomial(B=args.N_data, N_grid=args.N_grid)
        test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy = RFF(B=args.N_data, N_grid=args.N_grid)
    elif args.order == 1 and args.dataset == "poly":
        train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy = polynomial1(B=args.N_data, N_grid=args.N_grid)
        test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy = RFF1(B=args.N_data, N_grid=args.N_grid)
    elif args.order == 0 and args.dataset == "sin":
        train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy = RFF(B=args.N_data, N_grid=args.N_grid)
        test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy = polynomial(B=args.N_data, N_grid=args.N_grid)
    elif args.order == 1 and args.dataset == "sin":
        train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy = RFF1(B=args.N_data, N_grid=args.N_grid)
        test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy = polynomial1(B=args.N_data, N_grid=args.N_grid)
    @hk.transform
    def network(x, n, nabla_n):
        temp = MLP([64, 64, 64, 1])#FNO1d(16, 64) 
        return temp(x, n, nabla_n)
    net = hk.without_apply_rng(network)

    print(train_n.shape, train_nabla_n.shape, train_nabla2_n.shape, train_m.shape, train_y.shape, train_dy.shape, test_n.shape, test_nabla_n.shape, test_nabla2_n.shape, test_m.shape, test_y.shape, test_dy.shape)

    train_n, train_nabla_n, train_nabla2_n, train_m, train_y, train_dy, test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy = \
        train_n.reshape(-1), train_nabla_n.reshape(-1), train_nabla2_n.reshape(-1), train_m.reshape(-1), train_y.reshape(-1), train_dy.reshape(-1), test_n.reshape(-1), test_nabla_n.reshape(-1), test_nabla2_n.reshape(-1), test_m.reshape(-1), test_y.reshape(-1), test_dy.reshape(-1)
        #train_n.reshape(-1, 1), train_nabla_n.reshape(-1, 1), train_nabla2_n.reshape(-1, 1), train_m.reshape(-1, 1), train_y.reshape(-1, 1), train_dy.reshape(-1, 1), test_n.reshape(-1, 1), test_nabla_n.reshape(-1, 1), test_nabla2_n.reshape(-1, 1), test_m.reshape(-1, 1), test_y.reshape(-1, 1), test_dy.reshape(-1, 1)

    x = get_grid(jnp.shape(train_n)[0] // args.N_grid, args.N_grid)
    params = net.init(key, x[0], train_n[0], train_nabla_n[0])
    net_pred_fn = jax.vmap(net.apply, (None, 0, 0, 0))

    def fd_pred_fn(params, x, n, nabla_n, nabla2_n):
        y_n = jax.grad(net.apply, argnums=2)(params, x, n, nabla_n)
        temp1 = jax.grad(jax.grad(net.apply, argnums=3), argnums=1)(params, x, n, nabla_n)
        temp2 = jax.grad(jax.grad(net.apply, argnums=3), argnums=2)(params, x, n, nabla_n)
        temp3 = jax.grad(jax.grad(net.apply, argnums=3), argnums=3)(params, x, n, nabla_n)
        return y_n - temp1 - temp2 * nabla_n - temp3 * nabla2_n
    fd_pred_fn = jax.vmap(fd_pred_fn, (None, 0, 0, 0, 0))
    out = fd_pred_fn(params, get_grid(jnp.shape(test_n)[0] // args.N_grid, args.N_grid), test_n, test_nabla_n, test_nabla2_n)
    print(jnp.shape(out))

    optimizer = optax.adam(1e-3)
    #lr = optax.exponential_decay(init_value=1e-3, transition_steps=500, decay_rate=0.9)
    #optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def get_loss(params, x, n, nabla_n, nabla2_n, m, dy):
        y_pred = net_pred_fn(params, x, n, nabla_n)
        ret = jnp.mean((y_pred - m)**2)
        if args.lam_f > 0:
            f_pred = fd_pred_fn(params, x, n, nabla_n, nabla2_n)
            ret += args.lam_f * jnp.mean((f_pred - dy)**2)
        return ret

    @jax.jit
    def test_loss(params, x, n, nabla_n, nabla2_n, m, y, dy):
        y_pred = net_pred_fn(params, x, n, nabla_n)
        i_pred = jnp.mean(jnp.reshape(y_pred, (-1, args.N_grid)), axis=1)
        f_pred = fd_pred_fn(params, x, n, nabla_n, nabla2_n)
        return y_pred, i_pred, f_pred
        #err_func = jnp.linalg.norm(jnp.reshape(y_pred - m, (-1,))) / jnp.linalg.norm(jnp.reshape(m, (-1,)))
        #err_intor = jnp.linalg.norm(jnp.reshape(i_pred - y, (-1,))) / jnp.linalg.norm(jnp.reshape(y, (-1,)))
        #err_FD = jnp.linalg.norm(jnp.reshape(f_pred - dy, (-1,))) / jnp.linalg.norm(jnp.reshape(dy, (-1,)))
        #return err_func, err_intor, err_FD

    @jax.jit
    def step(params, opt_state, x, n, nabla_n, nabla2_n, m, dy):
        current_loss, gradients = jax.value_and_grad(get_loss)(params, x, n, nabla_n, nabla2_n, m, dy)
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state
    
    for epoch in tqdm(range(args.epochs)):
        x = get_grid(jnp.shape(train_n)[0] // args.N_grid, args.N_grid)
        current_loss, params, opt_state = step(params, opt_state, x, train_n, train_nabla_n, train_nabla2_n, train_m, train_dy)
        if epoch%1000==0:
            x = get_grid(jnp.shape(test_n)[0] // args.N_grid, args.N_grid)
            pred, pred_i, pred_f = test_loss(params, x, test_n, test_nabla_n, test_nabla2_n, test_m, test_y, test_dy)
            err_func = jnp.linalg.norm(jnp.reshape(pred - test_m, (-1,))) / jnp.linalg.norm(jnp.reshape(test_m, (-1,)))
            err_intor = jnp.linalg.norm(jnp.reshape(pred_i - test_y, (-1,))) / jnp.linalg.norm(jnp.reshape(test_y, (-1,)))
            err_FD = jnp.linalg.norm(jnp.reshape(pred_f - test_dy, (-1,))) / jnp.linalg.norm(jnp.reshape(test_dy, (-1,)))
            print('epoch %d, loss: %.3E, err_func: %.3E, err_intor: %.3E, err_FD: %.3E'%(epoch, current_loss, err_func, err_intor, err_FD))
