"""
Game costs.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil, Ellis Ratner)
"""

from functools import partial
from jax import jit, lax, jacfwd, hessian
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp

#from opinion_dynamics.utils import softmax


class Cost(object):
  """
  Base class for all cost functions. Structured as a functor so that it can be
  treated like a function, but support inheritance.
  """

  def __init__(self, name="", horizon=None, x_dim=None, ui_dim=None):
    self._name = name
    self._horizon = horizon
    self._x_dim = x_dim
    self._ui_dim = ui_dim

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl, ui: ArrayImpl, k: int = 0) -> ArrayImpl:
    """
    Abstract method.
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl): concatenated state of all subsystems (nx,)
        ui (ArrayImpl): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """
    raise NotImplementedError("get_cost is not implemented.")

  @partial(jit, static_argnums=(0,))
  def get_traj_cost(self, x: ArrayImpl, ui: ArrayImpl) -> ArrayImpl:
    """
    Evaluates this cost function along the given input state and/or control
    trajectory.

    Args:
        x (ArrayImpl): concatenated state of all subsystems (nx, N)
        ui (ArrayImpl): control of the subsystem (nui, N)

    Returns:
        ArrayImpl: costs (N,)
    """

    @jit
    def get_traj_cost_looper(k, costs):
      costs = costs.at[k].set(self.get_cost(x[:, k], ui[:, k], k))
      return costs

    costs = jnp.zeros(self._horizon)
    costs = lax.fori_loop(0, self._horizon, get_traj_cost_looper, costs)
    return costs

  @partial(jit, static_argnums=(0,))
  def get_grad(self, x: ArrayImpl, ui: ArrayImpl, k: int = 0) -> ArrayImpl:
    """
    Calculates the gradient w.r.t. state x, i.e. dc/dx, at (x, ui, k), and
    the gradient w.r.t. control ui, i.e. dc/dui, at (x, ui, k).

    Args:
        x (ArrayImpl): concatenated state of all subsystems (nx,)
        ui (ArrayImpl): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: gradient dc/dx (nx,)
        ArrayImpl: gradient dc/dui (nui,)
    """
    _gradients = jacfwd(self.get_cost, argnums=[0, 1])
    return _gradients(x, ui, k)

  @partial(jit, static_argnums=(0,))
  def get_traj_grad(self, x: ArrayImpl, ui: ArrayImpl) -> ArrayImpl:
    """
    Calculates gradients along the given input state and/or control trajectory.

    Args:
        x (ArrayImpl): concatenated state of all subsystems (nx, N)
        ui (ArrayImpl): control of the subsystem (nui, N)

    Returns:
        ArrayImpl: gradient dc/dx (nx, N)
        ArrayImpl: gradient dc/dui (nui, N)
    """

    @jit
    def get_traj_grad_looper(k, _carry):
      lxs, lus = _carry
      lx_k, lu_k = self.get_grad(x[:, k], ui[:, k], k)
      lxs = lxs.at[:, k].set(lx_k)
      lus = lus.at[:, k].set(lu_k)
      return lxs, lus

    lxs = jnp.zeros((self._x_dim, self._horizon))
    lus = jnp.zeros((self._ui_dim, self._horizon))
    lxs, lus = lax.fori_loop(0, self._horizon, get_traj_grad_looper, (lxs, lus))
    return lxs, lus

  @partial(jit, static_argnums=(0,))
  def get_hess(self, x: ArrayImpl, ui: ArrayImpl, k: int = 0) -> ArrayImpl:
    """
    Calculates the Hessians w.r.t. state x and control ui at (x, ui, k).

    Args:
        x (ArrayImpl): concatenated state of all subsystems (nx,)
        ui (ArrayImpl): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: Hessian w.r.t. x (nx, nx)
        ArrayImpl: Hessian w.r.t. ui (nui, nui)
    """
    _Hxx = hessian(self.get_cost, argnums=0)
    _Huu = hessian(self.get_cost, argnums=1)
    return _Hxx(x, ui, k), _Huu(x, ui, k)

  @partial(jit, static_argnums=(0,))
  def get_traj_hess(self, x: ArrayImpl, ui: ArrayImpl) -> ArrayImpl:
    """
    Calculates Hessians along the given input state and/or control trajectory.

    Args:
        x (ArrayImpl): concatenated state of all subsystems (nx, N)
        ui (ArrayImpl): control of the subsystem (nui, N)

    Returns:
        ArrayImpl: Hessian w.r.t. x (nx, nx, N)
        ArrayImpl: Hessian w.r.t. ui (nui, nui, N)
    """

    @jit
    def get_traj_hess_looper(k, _carry):
      Hxxs, Huus = _carry
      Hxx_k, Huu_k = self.get_hess(x[:, k], ui[:, k], k)
      Hxxs = Hxxs.at[:, :, k].set(Hxx_k)
      Huus = Huus.at[:, :, k].set(Huu_k)
      return Hxxs, Huus

    Hxxs = jnp.zeros((self._x_dim, self._x_dim, self._horizon))
    Huus = jnp.zeros((self._ui_dim, self._ui_dim, self._horizon))
    Hxxs, Huus = lax.fori_loop(0, self._horizon, get_traj_hess_looper, (Hxxs, Huus))
    return Hxxs, Huus


class ObstacleCost(Cost):

  def __init__(
      self, position_indices, point, max_distance, name="", horizon=None, x_dim=None, ui_dim=None
  ):
    """
    Obstacle cost, derived from Cost base class. Implements a cost function
    that depends only on state and penalizes min(0, dist - max_distance)^2.

    Initialize with dimension to add cost to and a max distance beyond
    which we impose no additional cost.
    """
    self._x_index, self._y_index = position_indices
    self._point = point
    self._max_distance = max_distance
    super(ObstacleCost, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl): concatenated state of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """
    # Computes the relative distance.
    dx = x[self._x_index] - self._point.x
    dy = x[self._y_index] - self._point.y
    relative_distance = jnp.sqrt(dx*dx + dy*dy)

    return jnp.minimum(relative_distance - self._max_distance, 0.)**2


class ProductStateProximityCostTwoPlayer(Cost):

  def __init__(
      self, position_indices, max_distance, name="", horizon=None, x_dim=None, ui_dim=None
  ):
    """
    Proximity cost for state spaces that are Cartesian products of individual
    systems' state spaces. Penalizes
      ``` sum_{i \ne j} min(distance(i, j) - max_distance, 0)^2 ```
    for all players i, j.

    For jit compatibility, number of players is hardcoded to 2 to avoid loops.
    """
    self._position_indices = position_indices
    self._max_distance = max_distance
    self._num_players = len(position_indices)
    self._max_exp_bound = 1e5
    super(ProductStateProximityCostTwoPlayer, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl): concatenated state vector of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """
    total_cost = 0.
    jsqrt = jnp.sqrt

    # Players' positional state indices.
    x1_idx, y1_idx = self._position_indices[0]
    x2_idx, y2_idx = self._position_indices[1]

    # Player 1.
    # -> Relative distance to Player 2.
    rel_dist = jsqrt((x[x1_idx] - x[x2_idx])**2 + (x[y1_idx] - x[y2_idx])**2)
    # total_cost += jnp.minimum(rel_dist - self._max_distance, 0.)**2
    total_cost += jnp.minimum(
        jnp.exp(jnp.minimum(rel_dist - self._max_distance, 0.)**2), self._max_exp_bound
    )

    # Player 2.
    # -> Relative distance to Player 1.
    rel_dist = jsqrt((x[x2_idx] - x[x1_idx])**2 + (x[y2_idx] - x[y1_idx])**2)
    # total_cost += jnp.minimum(rel_dist - self._max_distance, 0.)**2
    total_cost += jnp.minimum(
        jnp.exp(jnp.minimum(rel_dist - self._max_distance, 0.)**2), self._max_exp_bound
    )

    return total_cost


class ProductStateProximityCostThreePlayer(Cost):

  def __init__(
      self, position_indices, max_distance, name="", horizon=None, x_dim=None, ui_dim=None
  ):
    """
    Proximity cost for state spaces that are Cartesian products of individual
    systems' state spaces. Penalizes
      ``` sum_{i \ne j} min(distance(i, j) - max_distance, 0)^2 ```
    for all players i, j.

    Initialize with dimension to add cost to and threshold BELOW which
    to impose quadratic cost.

    For jit compatibility, number of players is hardcoded to 3 to avoid loops.
    """
    self._position_indices = position_indices
    self._max_distance = max_distance
    self._num_players = len(position_indices)
    super(ProductStateProximityCostThreePlayer, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl): concatenated state vector of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """
    total_cost = 0.
    jsqrt = jnp.sqrt

    # Players' positional state indices.
    x0_idx, y0_idx = self._position_indices[0]
    x1_idx, y1_idx = self._position_indices[1]
    x2_idx, y2_idx = self._position_indices[2]

    # Player 0.
    # -> Relative distance to Player 1.
    rel_dist = jsqrt((x[x0_idx] - x[x1_idx])**2 + (x[y0_idx] - x[y1_idx])**2)
    total_cost += jnp.minimum(rel_dist - self._max_distance, 0.)**2
    # -> Relative distance to Player 2.
    rel_dist = jsqrt((x[x0_idx] - x[x2_idx])**2 + (x[y0_idx] - x[y2_idx])**2)
    total_cost += jnp.minimum(rel_dist - self._max_distance, 0.)**2

    # Player 1.
    # -> Relative distance to Player 0.
    rel_dist = jsqrt((x[x1_idx] - x[x0_idx])**2 + (x[y1_idx] - x[y0_idx])**2)
    total_cost += jnp.minimum(rel_dist - self._max_distance, 0.)**2
    # -> Relative distance to Player 2.
    rel_dist = jsqrt((x[x1_idx] - x[x2_idx])**2 + (x[y1_idx] - x[y2_idx])**2)
    total_cost += jnp.minimum(rel_dist - self._max_distance, 0.)**2

    # Player 2.
    # -> Relative distance to Player 0.
    rel_dist = jsqrt((x[x2_idx] - x[x0_idx])**2 + (x[y2_idx] - x[y0_idx])**2)
    total_cost += jnp.minimum(rel_dist - self._max_distance, 0.)**2
    # -> Relative distance to Player 1.
    rel_dist = jsqrt((x[x2_idx] - x[x1_idx])**2 + (x[y2_idx] - x[y1_idx])**2)
    total_cost += jnp.minimum(rel_dist - self._max_distance, 0.)**2

    return total_cost


class ProximityCost(Cost):

  def __init__(
      self, position_indices, point_px, point_py, max_distance, name="", horizon=None, x_dim=None,
      ui_dim=None
  ):
    """
    Proximity cost, derived from Cost base class. Implements a cost function
    that depends only on state and penalizes -min(distance, max_distance)^2.
    """
    self._x_index, self._y_index = position_indices
    self._point_px = point_px
    self._point_py = point_py
    self._max_distance = max_distance
    super(ProximityCost, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl): concatenated state of the two systems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """

    dx = x[self._x_index] - self._point_px
    dy = x[self._y_index] - self._point_py
    rel_dist = jnp.sqrt(dx**2 + dy**2)
    return jnp.exp(jnp.minimum(rel_dist - self._max_distance, 0.)**2)


class QuadraticCost(Cost):

  def __init__(self, dimension, origin, is_x=False, name="", horizon=None, x_dim=None, ui_dim=None):
    """
    Quadratic cost, derived from Cost base class.

    Initialize with dimension to add cost to and origin to center the
    quadratic cost about.

    :param dimension: dimension to add cost
    :type dimension: uint
    :param threshold: value along the specified dim where the cost is zero
    :type threshold: float
    """
    self._dimension = dimension
    self._origin = origin
    self._is_x = is_x
    super(QuadraticCost, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl = None, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl, optional): concatenated state of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """
    if self._is_x:
      return (x[self._dimension] - self._origin)**2
    else:
      return (ui[self._dimension] - self._origin)**2


class QuadraticPolylineCost(Cost):

  def __init__(self, polyline, position_indices, name="", horizon=None, x_dim=None, ui_dim=None):
    """
    Quadratic cost that penalizes distance from a polyline.

    Initialize with a polyline.

    :param polyline: piecewise linear path which defines signed distances
    :type polyline: Polyline
    :param position_indices: indices of input corresponding to (x, y)
    :type position_indices: (uint, uint)
    """
    self._polyline = polyline
    self._x_index, self._y_index = position_indices
    super(QuadraticPolylineCost, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl): concatenated state of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """
    point = jnp.array((x[self._x_index], x[self._y_index]))
    signed_distance = self._polyline.signed_distance_to(point)
    return signed_distance**2


class ReferenceDeviationCost(Cost):

  def __init__(
      self, reference, dimension=None, is_x=False, name="", horizon=None, x_dim=None, ui_dim=None
  ):
    """
    Reference trajectory following cost. Penalizes sum of squared deviations
    from a reference trajectory (of a given quantity, e.g. x or u1 or u2).
    """
    self.reference = reference
    self._dimension = dimension
    self._is_x = is_x
    super(ReferenceDeviationCost, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl = None, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl, optional): concatenated state of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """
    if self._is_x:
      return (x[self._dimension] - self.reference)**2
    else:
      return (ui[self._dimension] - self.reference)**2


class ReferenceDeviationCostPxDependent(Cost):

  def __init__(
      self, reference, px_lb, px_dim, dimension=None, name="", horizon=None, x_dim=None, ui_dim=None
  ):
    """
    Reference trajectory following cost, dependent on px.
    """
    self.reference = reference
    self._dimension = dimension
    self._px_lb = px_lb
    self._px_dim = px_dim
    super(ReferenceDeviationCostPxDependent, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl = None, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl, optional): concatenated state of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """

    def true_fn(x):
      return (x[self._dimension] - self.reference)**2

    def false_fn(x):
      return 0.

    return lax.cond(x[self._px_dim] >= self._px_lb, true_fn, false_fn, x)


class OpnWeightedReferenceDeviationCost(Cost):

  def __init__(
      self, reference, dimension=None, is_x=False, name="", horizon=None, x_dim=None, ui_dim=None,
      z_idx: int = None, opn_idx: int = None, player_id: int = None
  ):
    """
    Opinion-weighted reference trajectory following cost.
    Penalizes sum of squared deviations from a reference trajectory (of a given
    quantity, e.g. x or u1 or u2).
    """
    self.reference = reference
    self._dimension = dimension
    self._is_x = is_x
    self._z_idx = z_idx
    self._opn_idx = opn_idx
    self._player_id = player_id
    super(OpnWeightedReferenceDeviationCost, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl = None, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl, optional): concatenated state of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """

    # This player's opinion state vector.
    zi = x[self._z_idx]

    # This player's option.
    weight_z = softmax(zi, self._opn_idx)

    if self._is_x:
      return weight_z * (x[self._dimension] - self.reference)**2
    else:
      return weight_z * (ui[self._dimension] - self.reference)**2


class SemiquadraticCost(Cost):

  def __init__(
      self, dimension, threshold, oriented_right, is_x=False, name="", horizon=None, x_dim=None,
      ui_dim=None
  ):
    """
    Semiquadratic cost, derived from Cost base class. Implements a
    cost function that is flat below a threshold and quadratic above, in the
    given dimension.

    Initialize with dimension to add cost to and threshold above which
    to impose quadratic cost.

    Args:
        dimension (int): dimension to add cost
        threshold (float): value above which to impose quadratic cost
        oriented_right (bool): Boolean flag determining which side of threshold
          to penalize
    """
    self._dimension = dimension
    self._threshold = threshold
    self._oriented_right = oriented_right
    self._is_x = is_x
    super(SemiquadraticCost, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl = None, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl, optional): concatenated state of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """
    if self._is_x:
      z = x
    else:
      z = ui

    def true_fn(z):
      # return (z[self._dimension] - self._threshold)**2
      return jnp.exp((z[self._dimension] - self._threshold)**2)

    def false_fn(z):
      return 0.

    if self._oriented_right:
      return lax.cond(z[self._dimension] > self._threshold, true_fn, false_fn, z)
    else:
      return lax.cond(z[self._dimension] < self._threshold, true_fn, false_fn, z)


class MaxVelCostPxDependent(Cost):

  def __init__(
      self, v_index, px_index, max_v, px_lb, px_ub, name="", horizon=None, x_dim=None, ui_dim=None
  ):
    """
    Penalize max velocity depending on the environment.
    """
    self._v_index = v_index
    self._px_index = px_index
    self._max_v = max_v
    self._px_lb = px_lb
    self._px_ub = px_ub
    self._max_exp_bound = 1e3
    super(MaxVelCostPxDependent, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl = None, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl, optional): concatenated state of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """

    def pred(x):
      px = x[self._px_index]
      v = x[self._v_index]
      return (v > self._max_v) & (self._px_lb < px) & (px < self._px_ub)

    def true_fn(x):
      # return (x[self._v_index] - self._max_v)**2
      return jnp.minimum(jnp.exp((x[self._v_index] - self._max_v)**2), self._max_exp_bound)

    def false_fn(x):
      return 0.

    return lax.cond(pred(x), true_fn, false_fn, x)


class SemiquadraticPolylineCost(Cost):

  def __init__(
      self, polyline, distance_threshold, position_indices, name="", horizon=None, x_dim=None,
      ui_dim=None
  ):
    """
    Semiquadratic cost that takes effect a fixed distance away from a Polyline.

    Initialize with a polyline, a threshold in distance from the polyline.

    Args:
        polyline (Polyline): piecewise linear path defining signed distances
        distance_threshold (float): value above which to penalize
        position_indices (int, int): indices of input corresponding to (x, y)
    """
    self._polyline = polyline
    self._distance_threshold = distance_threshold
    self._x_index, self._y_index = position_indices
    super(SemiquadraticPolylineCost, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl, optional): concatenated state of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """

    def true_fn(abs_signed_distance):
      return (abs_signed_distance - self._distance_threshold)**2

    def false_fn(abs_signed_distance):
      return 0.

    point = jnp.array((x[self._x_index], x[self._y_index]))
    signed_distance = self._polyline.signed_distance_to(point)
    abs_signed_distance = jnp.abs(signed_distance)

    return lax.cond(
        abs_signed_distance > self._distance_threshold, true_fn, false_fn, abs_signed_distance
    )


class BoxInputConstraintCost(Cost):

  def __init__(
      self, u_index, control_min, control_max, q1=1., q2=5., name="", horizon=None, x_dim=None,
      ui_dim=None
  ):
    """
    Box input constraint cost.
    """
    self._u_index = u_index
    self._control_min = control_min
    self._control_max = control_max
    self._q1 = q1
    self._q2 = q2
    super(BoxInputConstraintCost, self).__init__(name, horizon, x_dim, ui_dim)

  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl, ui: ArrayImpl = None, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (ArrayImpl): concatenated state vector of all subsystems (nx,)
        ui (ArrayImpl, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: scalar value of cost (scalar)
    """

    control = ui[self._u_index]
    margin_ub = control - self._control_max
    margin_lb = self._control_min - control

    c_ub = self._q1 * (jnp.exp(self._q2 * margin_ub) - 1.)
    c_lb = self._q1 * (jnp.exp(self._q2 * margin_lb) - 1.)
    return c_lb + c_ub

class LunarLanderCost(Cost):
    def __init__(self, u_index, xf_est, yf_est, name="", horizon=None, x_dim=None, ui_dim=None):
        """
        :param u_index: which element of ui this player uses (if ui_dim > 1)
        :param xf_est: desired/target x-position
        :param yf_est: desired/target y-position
        """
        super(LunarLanderCost, self).__init__(name, horizon, x_dim, ui_dim)
        self._u_index = u_index
        self._xf_est = xf_est
        self._yf_est = yf_est

    @partial(jit, static_argnums=(0,))
    def get_cost(self, x, ui, k=0):
        """
        x:  shape (6,) for the lander: [ px, py, angle, vx, vy, omega ]
        ui: shape (ui_dim,) for this player's control inputs
        k:  current time step index (unused here, but in the signature)
        """
        # If there is more than one control dimension, pick the index for this cost:
        u = ui[self._u_index]

        return (10.0 * (x[0] - self._xf_est)**2 +
                10.0 * (x[1] - self._yf_est)**2 +
                10.0 * x[3]**2 +
                10.0 * x[4]**2 +
                10.0 * (x[2] - jnp.pi / 2)**2 +
                10.0 * x[5]**2 +
                 1.0 * u**2)


  

