from typing import Callable, Any
import functools
import operator

import jax
import jax.numpy as jnp
from jax import flatten_util, tree_util

Pytree = Any

def flatten_output(afunc, unravel_first_arg):
    @functools.wraps(afunc)
    def _afunc(*args):
        x, *args = args
        return jax.flatten_util.ravel_pytree(afunc(unravel_first_arg(x), *args))[0]
    return _afunc

def conjugate_gradient(f : Callable, z_guess : Pytree) -> Pytree :
    # solves root finding problem : f(z) = 0 using newtons method. 
    # Every linear solve uses conjugate gradient method and therefore only uses hvp/vhp

    z_guess, unravel = flatten_util.ravel_pytree(z_guess)
    f_root = flatten_output(f, unravel_first_arg = unravel)
    vjp_f_root = lambda tangents, primals : jax.vjp(f_root, primals)[-1](tangents)[0]
    
    def body_fun(val):
        dval, _ = jax.scipy.sparse.linalg.cg(functools.partial(vjp_f_root, primals = val), f_root(val))
        return val - dval
    
    def cond_fun(val):
        return jnp.linalg.norm(f_root(val)) > 1e-5
    
    def scan_fun(carry, xs):
        carry = jax.lax.cond(cond_fun(carry), body_fun, lambda carry : carry, carry)
        return carry, None

    z, _ = jax.lax.scan(scan_fun, z_guess, xs = None, length = 20.)
    return unravel(z)

def newton_method(f : Callable, z_guess : Pytree) -> Pytree :
    # solves root finding problem : f(z) = 0 using newtons method 
    # Every linear solve uses explicit hessian 

    z_guess, unravel = flatten_util.ravel_pytree(z_guess)
    f_root = flatten_output(f, unravel_first_arg = unravel)
    grad_f_root = jax.jacrev(f_root)
    
    def body_fun(val):
        # conjugate gradient cannot be used because f is not forward mode differentiable and Hvp (used in cg) can be efficiently calculated using forward mode 
        dval = jnp.linalg.solve(grad_f_root(val), f_root(val))
        return val - dval
    
    def cond_fun(val):
        return jnp.linalg.norm(f_root(val)) > 1e-5
    
    def scan_fun(carry, xs):
        carry = jax.lax.cond(cond_fun(carry), body_fun, lambda carry : carry, carry)
        return carry, None

    z, _ = jax.lax.scan(scan_fun, z_guess, xs = None, length = 20.)
    return unravel(z)


@functools.partial(jax.custom_vjp, nondiff_argnums = (0, 1))
def root_finding_rev(solver : Callable, f : Callable, z : Pytree, p : Pytree) -> Pytree :
    # Reverse mode auto diff compatible root finding problem with f : Rn -> Rn
    _f = lambda z : f(z, p)
    return solver(_f, z)

def _root_finding_rev_fwd(solver, f, z, p):
    z_star = root_finding_rev(solver, f, z, p)
    return z_star, (z_star, p)

def _root_finding_rev_bwd(solver, f, res, gdot):
    
    z_star, p = res
    _, vjp_x = jax.vjp(lambda x : f(x, p), z_star)
    _, vjp_p = jax.vjp(lambda p : f(z_star, p), p)
    
    return None, *vjp_p(solver(lambda x : tree_util.tree_map(operator.add, vjp_x(x)[0], gdot), tree_util.tree_map(jnp.zeros_like, z_star)))

root_finding_rev.defvjp(_root_finding_rev_fwd, _root_finding_rev_bwd)


@functools.partial(jax.custom_jvp, nondiff_argnums = (0, 1))
def root_finding(solver : Callable, f : Callable, z : Pytree, p : Pytree) -> Pytree :
    # Forward mode auto diff compatible root finding problem with f : Rn -> Rn
    return solver(lambda z : f(z, p), z)

def _root_finding_fwd(solver, f, primals, tangents):
    z, p = primals
    zdot, pdot = tangents
    
    zstar = root_finding(solver, f, z, p)
    tangents_out = solver(lambda v : jax.jvp(lambda z, p : f(z, p), (zstar, p), (v, pdot))[-1], jnp.zeros_like(zdot))
    return zstar, tangents_out

root_finding.defjvp(_root_finding_fwd)



