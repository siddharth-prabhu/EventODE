import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import tree_util, flatten_util
import jax.random as jrandom

from event_ode import odeint_event


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
assert jnp.isclose(event_times[permutation], event_times_permute).all()


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
gradient_x, gradient_t, gradient_p = jax.grad(objective, argnums = (0, 1, 2))(x, t, p) # stop gradient issues
print("Gradient reverse mode autodiff", f"gradient x : {gradient_x}", f"gradient t : {gradient_t}", f"gradient p : {gradient_p}", sep = "\n")
gradient_x, gradient_t, gradient_p = jax.jacfwd(objective, argnums = (0, 1, 2))(x, t, p)
print("Gradient reverse mode autodiff", f"gradient x : {gradient_x}", f"gradient t : {gradient_t}", f"gradient p : {gradient_p}", sep = "\n")
fd_x, fd_t, fd_p = gradient_fd(x, t, p, 1e-4)
print("Gradient finite difference", f"gradient x : {fd_x}", f"gradient t : {fd_t}", f"gradient p : {fd_p}", sep = "\n")

# Testing hessians
hess = jax.hessian(objective, argnums = (0, 1, 2))(x, t, p)
print("Hessian fwd(rev) : ", hess, sep = "\n")
hess_fwd = jax.jacfwd(jax.jacfwd(objective, argnums = (0, 1, 2)), argnums = (0, 1, 2))(x, t, p)
print("Hessian fwd(fwd) : ", hess_fwd, sep = "\n")
hess_rev = jax.jacrev(jax.jacrev(objective, argnums = (0, 1, 2)), argnums = (0, 1, 2))(x, t, p)
print("Hessian rev(rev) : ", hess_rev, sep = "\n")
