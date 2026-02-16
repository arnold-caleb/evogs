"""
Quaternion Utilities for Rotation Evolution

This module provides quaternion operations for geometrically correct
rotation integration on the SO(3) manifold.

Key Functions:
- quaternion_multiply: Hamilton product q1 ⊗ q2
- quaternion_derivative: Compute dq/dt from angular velocity ω
- angular_velocity_to_quaternion: Convert ω to rotation quaternion
"""

import torch
import torch.nn.functional as F


def quaternion_multiply(q1, q2):
    """
    Hamilton product for quaternions: q1 ⊗ q2
    
    Quaternion format: [w, x, y, z] where w is scalar part
    
    Formula:
        (w1, v1) ⊗ (w2, v2) = (w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2)
    
    Args:
        q1 (Tensor): [N, 4] first quaternions
        q2 (Tensor): [N, 4] second quaternions
    
    Returns:
        Tensor: [N, 4] product quaternions q1 ⊗ q2
    
    Example:
        >>> q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity
        >>> q2 = torch.tensor([[0.707, 0.0, 0.0, 0.707]])  # 90° around z
        >>> q_result = quaternion_multiply(q1, q2)
        >>> print(q_result)  # Should be q2
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    # Hamilton product formula
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=-1)


def quaternion_derivative(q, omega):
    """
    Compute quaternion time derivative from angular velocity.
    
    Formula:
        dq/dt = 0.5 * q ⊗ [0, ω]
    
    This is the fundamental equation of rotational kinematics.
    Integrating this ensures rotations stay on the SO(3) manifold.
    
    Args:
        q (Tensor): [N, 4] current quaternions
        omega (Tensor): [N, 3] angular velocity in body frame [ωx, ωy, ωz]
    
    Returns:
        Tensor: [N, 4] quaternion derivatives dq/dt
    
    Notes:
        - Angular velocity ω describes rotation rate around axis ||ω|| with
          speed ||ω|| rad/s
        - This formulation automatically maintains ||q|| = 1 during integration
        - No normalization needed after integration (theoretically)
    
    Example:
        >>> q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity rotation
        >>> omega = torch.tensor([[0.0, 0.0, 1.0]])  # 1 rad/s around z
        >>> dq_dt = quaternion_derivative(q, omega)
        >>> print(dq_dt)  # [0, 0, 0, 0.5] - rotating around z
    """
    N = q.shape[0]
    device = q.device
    
    # Convert angular velocity to pure quaternion [0, ωx, ωy, ωz]
    omega_quat = torch.cat([
        torch.zeros(N, 1, device=device),
        omega
    ], dim=-1)
    
    # dq/dt = 0.5 * q ⊗ [0, ω]
    dq_dt = 0.5 * quaternion_multiply(q, omega_quat)
    
    return dq_dt


def angular_velocity_to_quaternion(omega, dt):
    """
    Convert angular velocity to quaternion rotation via exponential map.
    
    This is useful for single-step integration:
        q(t+dt) = q(t) ⊗ exp(0.5 * dt * [0, ω])
    
    Args:
        omega (Tensor): [N, 3] angular velocity
        dt (float or Tensor): time step
    
    Returns:
        Tensor: [N, 4] quaternion representing rotation
    
    Formula:
        axis = ω / ||ω||
        angle = ||ω|| * dt
        q = [cos(angle/2), sin(angle/2) * axis]
    
    Example:
        >>> omega = torch.tensor([[0.0, 0.0, torch.pi]])  # 180° around z
        >>> dt = 1.0
        >>> q = angular_velocity_to_quaternion(omega, dt)
        >>> print(q)  # [0, 0, 0, 1] - 180° rotation around z
    """
    N = omega.shape[0]
    device = omega.device
    
    # Rotation angle (magnitude of angular velocity × time)
    angle = omega.norm(dim=-1, keepdim=True) * dt  # [N, 1]
    
    # Rotation axis (normalized angular velocity)
    # Add epsilon to avoid division by zero for zero angular velocity
    axis = omega / (omega.norm(dim=-1, keepdim=True) + 1e-8)  # [N, 3]
    
    # Axis-angle to quaternion
    # q = [cos(θ/2), sin(θ/2) * axis]
    half_angle = angle / 2
    w = torch.cos(half_angle)
    xyz = torch.sin(half_angle) * axis
    
    return torch.cat([w, xyz], dim=-1)


def quaternion_to_angular_velocity(q1, q2, dt):
    """
    Compute angular velocity from quaternion difference.
    
    Given two quaternions q1 and q2 separated by time dt,
    compute the angular velocity ω that rotates q1 to q2.
    
    Args:
        q1 (Tensor): [N, 4] start quaternions
        q2 (Tensor): [N, 4] end quaternions
        dt (float): time interval
    
    Returns:
        Tensor: [N, 3] angular velocity
    
    Formula:
        q_diff = q2 ⊗ q1^(-1)
        [w, v] = q_diff
        θ = 2 * atan2(||v||, w)
        axis = v / ||v||
        ω = (θ / dt) * axis
    """
    # Quaternion difference (relative rotation)
    q1_inv = torch.cat([q1[:, 0:1], -q1[:, 1:]], dim=-1)
    q_diff = quaternion_multiply(q2, q1_inv)
    
    # Extract axis-angle
    w = q_diff[:, 0:1]
    v = q_diff[:, 1:]
    
    # Angle from quaternion
    v_norm = v.norm(dim=-1, keepdim=True)
    theta = 2 * torch.atan2(v_norm, w)
    
    # Axis (normalized)
    axis = v / (v_norm + 1e-8)
    
    # Angular velocity = (angle / time) * axis
    omega = (theta / dt) * axis
    
    return omega


def normalize_quaternion(q):
    """
    Normalize quaternions to unit length.
    
    Args:
        q (Tensor): [N, 4] quaternions
    
    Returns:
        Tensor: [N, 4] normalized quaternions
    """
    return F.normalize(q, dim=-1)


def quaternion_geodesic_distance(q1, q2):
    """
    Compute geodesic distance between quaternions on SO(3).
    
    This is the angle of rotation needed to go from q1 to q2.
    
    Args:
        q1 (Tensor): [N, 4] first quaternions
        q2 (Tensor): [N, 4] second quaternions
    
    Returns:
        Tensor: [N] angular distances in radians
    
    Formula:
        distance = 2 * arccos(|<q1, q2>|)
    
    Range: [0, π]
    """
    # Inner product
    dot = (q1 * q2).sum(dim=-1).abs()
    
    # Clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Geodesic distance
    distance = 2 * torch.acos(dot)
    
    return distance


def quaternion_slerp(q1, q2, t):
    """
    Spherical linear interpolation between quaternions.
    
    Args:
        q1 (Tensor): [N, 4] start quaternions
        q2 (Tensor): [N, 4] end quaternions
        t (float or Tensor): interpolation parameter in [0, 1]
    
    Returns:
        Tensor: [N, 4] interpolated quaternions
    
    Notes:
        - t=0 returns q1
        - t=1 returns q2
        - Interpolation follows shortest path on SO(3)
    """
    # Compute cosine of angle between q1 and q2
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    
    # If dot < 0, negate q2 to take shortest path
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)
    
    # Clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Compute angle
    theta = torch.acos(dot)
    
    # Slerp formula
    sin_theta = torch.sin(theta)
    
    # Handle case where quaternions are very close
    if isinstance(t, float):
        t = torch.tensor(t, device=q1.device)
    
    # Avoid division by zero
    epsilon = 1e-8
    w1 = torch.sin((1.0 - t) * theta) / (sin_theta + epsilon)
    w2 = torch.sin(t * theta) / (sin_theta + epsilon)
    
    # Handle case where sin(theta) ≈ 0 (quaternions are very close)
    w1 = torch.where(sin_theta < epsilon, 1.0 - t, w1)
    w2 = torch.where(sin_theta < epsilon, t, w2)
    
    q_interp = w1 * q1 + w2 * q2
    
    return normalize_quaternion(q_interp)


# Batch operation wrappers
def batch_quaternion_multiply(q1, q2):
    """Alias for quaternion_multiply (for backward compatibility)"""
    return quaternion_multiply(q1, q2)


# Test utilities
def test_quaternion_operations():
    """
    Test quaternion operations for correctness.
    
    Returns:
        bool: True if all tests pass
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing quaternion operations...")
    
    # Test 1: Identity multiplication
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    q_test = torch.tensor([[0.707, 0.0, 0.0, 0.707]], device=device)
    q_result = quaternion_multiply(q_identity, q_test)
    assert torch.allclose(q_result, q_test, atol=1e-6), "Identity multiplication failed"
    print("✓ Identity multiplication")
    
    # Test 2: Quaternion derivative
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    omega = torch.tensor([[0.0, 0.0, 1.0]], device=device)  # 1 rad/s around z
    dq_dt = quaternion_derivative(q, omega)
    expected = torch.tensor([[0.0, 0.0, 0.0, 0.5]], device=device)
    assert torch.allclose(dq_dt, expected, atol=1e-6), "Quaternion derivative failed"
    print("✓ Quaternion derivative")
    
    # Test 3: Angular velocity to quaternion
    omega = torch.tensor([[0.0, 0.0, torch.pi]], device=device)
    q_rot = angular_velocity_to_quaternion(omega, dt=1.0)
    expected = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    assert torch.allclose(q_rot, expected, atol=1e-5), "Angular velocity conversion failed"
    print("✓ Angular velocity to quaternion")
    
    # Test 4: Normalization preserves direction
    q_unnorm = torch.tensor([[2.0, 0.0, 0.0, 0.0]], device=device)
    q_norm = normalize_quaternion(q_unnorm)
    assert torch.allclose(q_norm.norm(dim=-1), torch.ones(1, device=device), atol=1e-6)
    print("✓ Quaternion normalization")
    
    print("\n✅ All tests passed!")
    return True


if __name__ == '__main__':
    test_quaternion_operations()

