import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import tree_util, flatten_util
import jax.random as jrandom

from event_ode import odeint_event
from utils import root_finding_fwd

##############################################################################################################
# Examples of ode with events

# Function for single discretized point of x
def bar(x, t, p): return jnp.array([- p[0] * x[0], - p[0] * x[1]]) # Before event is triggered
def foo(x, t, p): return jnp.array([- p[0] * x[0]**2, - p[0] * x[1]**2]) # After event is triggered
def event(x, t): return jnp.array([x[0] - 1.]) # Event condition


# Functions vmapped over discretized points of x
def afunc(x, t, args):
    event_times, p = args
    return jax.vmap(lambda _x, _event : jax.lax.cond(t <= _event, lambda : bar(_x, t, p), lambda : foo(_x, t,  p)))(x, event_times)

def transfer(trans_func, x, t, args): return trans_func(afunc, event, x, t, args) # Transfer function for sensitivity transfer
def event_vmap(x, t): return jax.vmap(event, in_axes = (0, None))(x, t)


# sorted event times
time_span = jnp.arange(0, 5., 0.01)
p = jnp.array([2.])
xinit = jnp.arange(2, 10. * 2 + 2).reshape(-1, 2) # shape = (Discretized points X dimension of x)
trajectory, event_times = odeint_event(afunc, event_vmap, transfer, xinit, time_span, p)

# permuted event times
key = jrandom.PRNGKey(10)
permutation = jrandom.permutation(key, jnp.arange(xinit.shape[0]))
trajectory_permute, event_times_permute = odeint_event(afunc, event_vmap, transfer, xinit[permutation], time_span, p)
assert jnp.allclose(event_times[permutation], event_times_permute)


# checking automatic differentiation gradients with finite difference

def objective(x, t, p):
    time_span = jnp.linspace(0, t, 100)
    xinit = jnp.tile(x, (10, 1)) + jrandom.normal(key, shape = (10, 2)) * 0.1
    solution = odeint_event(afunc, event_vmap, transfer, xinit, time_span, p)
    return jnp.mean((solution[0])**2)


# check gradients using finite difference
def gradient_fd(x, t, p, eps):

    args_flatten, unravel = flatten_util.ravel_pytree(( x, t, p ))
    def _grad(v):
        loss = objective(*unravel(args_flatten + eps * v)) - objective(*unravel(args_flatten - eps * v))
        return loss / 2 / eps
    
    _gradients = jax.vmap(_grad)(jnp.eye(len(args_flatten)))
    return unravel(_gradients)


x = xinit[0]
t = jnp.array(5.)

# Testing gradients
gradient_x, gradient_t, gradient_p = jax.grad(objective, argnums = (0, 1, 2))(x, t, p) # reverse-mode autodiff compatible
gradient_x, gradient_t, gradient_p = jax.jacfwd(objective, argnums = (0, 1, 2))(x, t, p) # forward-mode autodiff compatible

eps = 1e-4
fd_x, fd_t, fd_p = gradient_fd(x, t, p, eps)

assert jnp.allclose(fd_x, gradient_x, atol = 100 * eps), "Finite difference does not match for x"
assert jnp.allclose(fd_p, gradient_p, atol = 100 * eps), "Finite difference does not match for p"
assert jnp.allclose(fd_t, gradient_t, atol = 100 * eps), "Finite difference does not match for t"

# Testing hessians - forward-over-reverse mode autodiff
hess = jax.hessian(objective, argnums = (0, 1, 2))(x, t, p)


##############################################################################################################
# Examples of root-finding problem

def f_root(x_guess, p):
    x = x_guess
    for _ in range(4):
        x = p @ jnp.tanh(x)
    return x_guess - x

def obj(p, save):
    x_opt = root_finding_fwd(f_root, jnp.array([0.5, 1.]), p, save)
    return jnp.mean(x_opt**2), x_opt


# Saving inverse is linear in the tangents and therefore fwd/rev can be computed
p = jnp.arange(1, 5., 1).reshape(2, -1)
_jacfwd_root_save, x_opt = jax.jacfwd(obj, has_aux = True)(p, True)
_jacrev_root_save, _ = jax.jacrev(obj, has_aux = True)(p, True)

_jacfwd_root, x_opt = jax.jacfwd(obj, has_aux = True)(p, False)
_jacrev_root, _ = jax.jacrev(obj, has_aux = True)(p, False)

# check gradients using finite difference
def gradient_fd(eps):
    args_flatten, unravel = flatten_util.ravel_pytree(p)
    _gradients = jax.vmap(lambda v : (obj(unravel(args_flatten + eps * v), True)[0] - obj(unravel(args_flatten - eps * v), True)[0]) / 2 / eps)(jnp.eye(len(args_flatten)))
    return unravel(_gradients)

eps = 1e-4
fd_p = gradient_fd(eps)
assert jnp.allclose(_jacfwd_root_save, _jacfwd_root), "Forward-mode autodiff are not equal"
assert jnp.allclose(_jacrev_root_save, _jacrev_root), "Reverse-mode autodiff are not equal"
assert jnp.allclose(fd_p, _jacfwd_root_save, atol = 100 * eps), "Finite difference derivatives dont match"


# Note - Higher order derivatives should be computed with reuse_inverse = False
p_flat, unravel = flatten_util.ravel_pytree(p)
hess = jax.hessian(lambda p : obj(unravel(p), False)[0])(p_flat) # Forward and reverse mode autodiff compatible

def hessian_fd(eps):
    _grad = jax.grad(lambda p : obj(unravel(p), False)[0])
    return jax.vmap(lambda v : (_grad(p_flat + eps * v) - _grad(p_flat - eps * v)) / 2 / eps)(jnp.eye(len(p_flat)))
    
hess_fd = hessian_fd(eps)
assert jnp.allclose(hess, hess_fd, atol = 100 * eps)