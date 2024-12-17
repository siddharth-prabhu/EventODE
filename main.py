import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
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