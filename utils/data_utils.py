"""
Data Structures and Transformation Utilities for NeSVoR2

This module combines functionality for:
1. Rigid transformations (rotation + translation) in 3D space
2. Medical image data structures (Image, Slice, Volume, Stack)
3. Coordinate system conversions (NIfTI affine ↔ internal format)
4. File I/O for medical imaging (NIfTI format)

Key Concepts:
-------------
Axis-Angle Representation:
    Rotation encoded as axis * angle, where:
    - Direction of vector = rotation axis
    - Magnitude of vector = rotation angle in radians
    - Compact: 3 parameters instead of 9 for matrix

Matrix Representation:
    3x4 matrix [R | t] where:
    - R (3x3) = rotation matrix
    - t (3x1) = translation vector

Translation Convention:
    - trans_first=True: Apply translation first, then rotate
    - trans_first=False: Apply rotation first, then translate

Medical Imaging Conventions:
    - Internal: (D, H, W) = (depth, height, width) = (z, y, x)
    - NIfTI file: (W, H, D) = (x, y, z) - requires transposition
"""

from __future__ import annotations
from typing import Tuple, Union, Optional, Dict, List, Sequence, cast, Iterable
import os
import nibabel as nib
import torch
import numpy as np
import torch.nn.functional as F
from .gen_utils import meshgrid, resample, DeviceType, PathType


# Small epsilon for numerical stability in rotation computations
TRANSFORM_EPS = 1e-6


# ============================================================================
# SECTION 1: AXIS-ANGLE ↔ MATRIX CONVERSIONS
# ============================================================================
# Convert between axis-angle and rotation matrix representations


def axisangle2mat(axisangle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix.

    Axis-angle (Rodrigues formula) is a compact rotation representation:
    - Rotation axis: direction of vector (nx, ny, nz)
    - Rotation angle: magnitude of vector θ = ||axisangle[:3]||

    For small angles, uses first-order Taylor approximation (skew-symmetric matrix).
    For large angles, uses Rodrigues' rotation formula.

    Args:
        axisangle: Tensor of shape (N, 6) where:
            - [:, :3]: Axis-angle rotation (rx, ry, rz)
            - [:, 3:]: Translation (tx, ty, tz)

    Returns:
        mat: Transformation matrix of shape (N, 3, 4) = [R | t]

    Note:
        Handles numerical instability near zero rotation angle using
        conditional computation based on TRANSFORM_EPS threshold.

    Reference:
        Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        where K is the skew-symmetric matrix of the normalized axis.
    """
    # Compute rotation angle
    theta2 = axisangle[:, :3].pow(2).sum(-1)  # θ²
    small_angle = theta2 <= TRANSFORM_EPS
    theta = torch.clamp(theta2, min=TRANSFORM_EPS).sqrt()  # θ (clamped for stability)

    # Normalize rotation axis
    ang_x = axisangle[:, 0] / theta
    ang_y = axisangle[:, 1] / theta
    ang_z = axisangle[:, 2] / theta

    # Precompute trig functions
    s = torch.sin(theta)
    c = torch.cos(theta)
    o_c = 1 - c  # (1 - cos(θ))

    # Large angle case: Full Rodrigues formula
    # R = I + sin(θ)K + (1-cos(θ))K²
    mat1 = torch.stack(
        (
            c + ang_x * ang_x * o_c,  # R[0,0]
            ang_x * ang_y * o_c - ang_z * s,  # R[0,1]
            ang_y * s + ang_x * ang_z * o_c,  # R[0,2]
            ang_z * s + ang_x * ang_y * o_c,  # R[1,0]
            c + ang_y * ang_y * o_c,  # R[1,1]
            -ang_x * s + ang_y * ang_z * o_c,  # R[1,2]
            -ang_y * s + ang_x * ang_z * o_c,  # R[2,0]
            ang_x * s + ang_y * ang_z * o_c,  # R[2,1]
            c + ang_z * ang_z * o_c,  # R[2,2]
        ),
        -1,
    )

    # Small angle case: First-order approximation (skew-symmetric)
    # R ≈ I + K for small θ, where K is skew-symmetric matrix
    ones = torch.ones_like(o_c)
    mat2 = torch.stack(
        (
            ones,  # R[0,0] = 1
            -axisangle[:, 2],  # R[0,1] = -rz
            axisangle[:, 1],  # R[0,2] = ry
            axisangle[:, 2],  # R[1,0] = rz
            ones,  # R[1,1] = 1
            -axisangle[:, 0],  # R[1,2] = -rx
            -axisangle[:, 1],  # R[2,0] = -ry
            axisangle[:, 0],  # R[2,1] = rx
            ones,  # R[2,2] = 1
        ),
        -1,
    )

    # Select appropriate formula based on angle magnitude
    mat = torch.where(small_angle[..., None], mat2, mat1).reshape((-1, 3, 3))

    # Append translation vector to create [R | t] matrix
    mat = torch.cat((mat, axisangle[:, -3:, None]), -1)

    return mat


def mat2axisangle(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to axis-angle representation.

    Uses quaternion as intermediate representation for numerical stability.
    The conversion handles four cases based on which diagonal element is largest,
    to avoid division by small numbers.

    Args:
        mat: Transformation matrix of shape (N, 3, 4) = [R | t]

    Returns:
        axisangle: Tensor of shape (N, 6) = [rx, ry, rz, tx, ty, tz]

    Algorithm:
        1. Matrix → Quaternion (4 cases based on diagonal dominance)
        2. Quaternion → Axis-angle
        3. Handle quaternion sign ambiguity (q and -q represent same rotation)

    Note:
        This is the inverse of axisangle2mat() and handles numerical
        edge cases carefully.
    """
    # Extract rotation matrix elements
    r00 = mat[:, 0, 0]
    r01 = mat[:, 0, 1]
    r02 = mat[:, 0, 2]
    r10 = mat[:, 1, 0]
    r11 = mat[:, 1, 1]
    r12 = mat[:, 1, 2]
    r20 = mat[:, 2, 0]
    r21 = mat[:, 2, 1]
    r22 = mat[:, 2, 2]

    # Determine which conversion case to use based on trace
    mask_d2 = r22 < TRANSFORM_EPS  # z-diagonal small
    mask_d0_d1 = r00 > r11  # x > y diagonal
    mask_d0_nd1 = r00 < -r11  # x < -y diagonal

    # Case 1: Normal case (trace is positive)
    # q = [w, x, y, z] where w = (1 + trace)/4
    s1 = 2 * torch.sqrt(torch.clamp(r00 + r11 + r22 + 1, min=TRANSFORM_EPS))
    w1 = s1 / 4
    x1 = (r21 - r12) / s1
    y1 = (r02 - r20) / s1
    z1 = (r10 - r01) / s1

    # Case 2: x-diagonal dominant
    s2 = 2 * torch.sqrt(torch.clamp(r00 - r11 - r22 + 1, min=TRANSFORM_EPS))
    w2 = (r21 - r12) / s2
    x2 = s2 / 4
    y2 = (r01 + r10) / s2
    z2 = (r02 + r20) / s2

    # Case 3: y-diagonal dominant
    s3 = 2 * torch.sqrt(torch.clamp(r11 - r00 - r22 + 1, min=TRANSFORM_EPS))
    w3 = (r02 - r20) / s3
    x3 = (r01 + r10) / s3
    y3 = s3 / 4
    z3 = (r12 + r21) / s3

    # Case 4: z-diagonal dominant
    s4 = 2 * torch.sqrt(torch.clamp(r22 - r00 - r11 + 1, min=TRANSFORM_EPS))
    w4 = (r10 - r01) / s4
    x4 = (r02 + r20) / s4
    y4 = (r12 + r21) / s4
    z4 = s4 / 4

    # Select appropriate case for each element in batch
    case1 = (~mask_d2) & ~(mask_d0_nd1)
    case2 = mask_d2 & mask_d0_d1
    case3 = mask_d2 & (~mask_d0_d1)

    w = torch.where(case1, w1, torch.where(case2, w2, torch.where(case3, w3, w4)))
    x = torch.where(case1, x1, torch.where(case2, x2, torch.where(case3, x3, x4)))
    y = torch.where(case1, y1, torch.where(case2, y2, torch.where(case3, y3, y4)))
    z = torch.where(case1, z1, torch.where(case2, z2, torch.where(case3, z3, z4)))

    # Ensure positive w (resolve quaternion sign ambiguity)
    neg_w = w < 0
    x = torch.where(neg_w, -x, x)
    y = torch.where(neg_w, -y, y)
    z = torch.where(neg_w, -z, z)
    w = torch.where(neg_w, -w, w)

    # Convert quaternion to axis-angle
    # θ = 2 * atan2(||v||, w) where v = [x, y, z]
    # axis-angle = θ * v / ||v||
    tmp = x * x + y * y + z * z
    si = torch.sqrt(torch.clamp(tmp, min=TRANSFORM_EPS))
    theta = 2 * torch.atan2(si, w)
    fac = torch.where(tmp > TRANSFORM_EPS, theta / si, 2.0 / w)

    x = x * fac
    y = y * fac
    z = z * fac

    # Concatenate rotation and translation
    axisangle = torch.cat((x[:, None], y[:, None], z[:, None], mat[:, :, -1]), -1)
    return axisangle


# ============================================================================
# SECTION 2: RIGID TRANSFORMATION CLASS
# ============================================================================


class RigidTransform(object):
    """
    Rigid transformation (rotation + translation) in 3D space.

    A rigid transformation preserves distances and angles. It can be represented as:
    - Axis-angle: 6 parameters [rx, ry, rz, tx, ty, tz] (compact)
    - Matrix: 3x4 matrix [R | t] where R is 3x3 rotation, t is 3x1 translation

    This class stores one representation and converts to the other on demand.
    Supports batch operations (N transformations simultaneously).

    Attributes:
        trans_first: If True, apply translation before rotation (default)
                    If False, apply rotation before translation
        _axisangle: Axis-angle representation (N, 6) or None
        _matrix: Matrix representation (N, 3, 4) or None

    Example:
        # Create from axis-angle
        ax = torch.tensor([[0.1, 0.2, 0.3, 5.0, 10.0, 15.0]])  # Small rotation + translation
        transform = RigidTransform(ax, trans_first=True)

        # Get matrix representation
        mat = transform.matrix()  # Returns (1, 3, 4)

        # Apply to points
        points = torch.randn(100, 3)
        transformed = transform_points(transform, points)

        # Invert transformation
        inv_transform = transform.inv()

        # Compose transformations: T3 = T1 ∘ T2
        composed = transform1.compose(transform2)
    """

    def __init__(
        self, data: torch.Tensor, trans_first: bool = True, device: DeviceType = None
    ) -> None:
        """
        Initialize rigid transformation.

        Args:
            data: Either:
                - Axis-angle: (N, 6) tensor [rx, ry, rz, tx, ty, tz]
                - Matrix: (N, 3, 4) tensor [R | t]
            trans_first: Translation convention (see class docstring)
            device: Device to move data to (optional)

        Raises:
            Exception: If data shape doesn't match either format
        """
        self.trans_first = trans_first
        self._axisangle = None
        self._matrix = None

        if device is not None:
            data = data.to(device)

        # Detect format from shape
        if data.shape[1] == 6 and data.ndim == 2:  # Axis-angle format
            self._axisangle = data
        elif data.shape[1] == 3 and data.ndim == 3:  # Matrix format
            self._matrix = data
        else:
            raise Exception("Unknown format for rigid transform!")

    def matrix(self, trans_first: bool = True) -> torch.Tensor:
        """
        Get matrix representation [R | t].

        Args:
            trans_first: Desired translation convention

        Returns:
            Tensor of shape (N, 3, 4)

        Note:
            Automatically converts from axis-angle if needed and handles
            translation convention conversion.
        """
        # Get matrix in stored convention
        if self._matrix is not None:
            mat = self._matrix
        elif self._axisangle is not None:
            mat = axisangle2mat(self._axisangle)
        else:
            raise ValueError("Both data are None!")

        # Convert convention if needed
        if self.trans_first == True and trans_first == False:
            mat = mat_first2last(mat)
        elif self.trans_first == False and trans_first == True:
            mat = mat_last2first(mat)

        return mat

    def axisangle(self, trans_first: bool = True) -> torch.Tensor:
        """
        Get axis-angle representation.

        Args:
            trans_first: Desired translation convention

        Returns:
            Tensor of shape (N, 6) = [rx, ry, rz, tx, ty, tz]

        Note:
            Automatically converts from matrix if needed and handles
            translation convention conversion.
        """
        # Get axis-angle in stored convention
        if self._axisangle is not None:
            ax = self._axisangle
        elif self._matrix is not None:
            ax = mat2axisangle(self._matrix)
        else:
            raise ValueError("Both data are None!")

        # Convert convention if needed
        if self.trans_first == True and trans_first == False:
            ax = ax_first2last(ax)
        elif self.trans_first == False and trans_first == True:
            ax = ax_last2first(ax)

        return ax

    def inv(self) -> "RigidTransform":
        """
        Compute inverse transformation.

        For rigid transform [R | t]:
        - Inverse is [R^T | -R^T @ t] (when trans_first=True)

        Returns:
            New RigidTransform representing the inverse

        Note:
            T ∘ T^(-1) = Identity
        """
        mat = self.matrix(trans_first=True)
        R = mat[:, :, :3]
        t = mat[:, :, 3:]

        # Inverse: R^T and -R^T @ t
        mat = torch.cat((R.transpose(-2, -1), -torch.matmul(R, t)), -1)
        return RigidTransform(mat, trans_first=True)

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        """
        Compose two transformations: result = self ∘ other.

        For transformations T1 and T2:
        - First apply T2, then apply T1
        - Result transforms as: T1(T2(x))

        Args:
            other: Second transformation (applied first)

        Returns:
            Composed transformation

        Note:
            Composition is associative but NOT commutative:
            T1 ∘ T2 ≠ T2 ∘ T1 in general
        """
        mat1 = self.matrix(trans_first=True)
        mat2 = other.matrix(trans_first=True)

        R1 = mat1[:, :, :3]
        t1 = mat1[:, :, 3:]
        R2 = mat2[:, :, :3]
        t2 = mat2[:, :, 3:]

        # Compose: R = R1 @ R2, t = t2 + R2^T @ t1
        R = torch.matmul(R1, R2)
        t = t2 + torch.matmul(R2.transpose(-2, -1), t1)

        mat = torch.cat((R, t), -1)
        return RigidTransform(mat, trans_first=True)

    def __getitem__(self, idx) -> "RigidTransform":
        """
        Index into batch of transformations.

        Args:
            idx: Integer, slice, or boolean mask

        Returns:
            New RigidTransform with selected elements
        """
        if self._axisangle is not None:
            data = self._axisangle[idx]
            if len(data.shape) < 2:
                data = data.unsqueeze(0)  # Ensure batch dimension
        elif self._matrix is not None:
            data = self._matrix[idx]
            if len(data.shape) < 3:
                data = data.unsqueeze(0)
        else:
            raise Exception("Both data are None!")

        return RigidTransform(data, self.trans_first)

    def detach(self) -> "RigidTransform":
        """
        Detach from computation graph (no gradients).

        Returns:
            New RigidTransform with detached data
        """
        if self._axisangle is not None:
            data = self._axisangle.detach()
        elif self._matrix is not None:
            data = self._matrix.detach()
        else:
            raise Exception("Both data are None!")

        return RigidTransform(data, self.trans_first)

    def clone(self) -> "RigidTransform":
        """
        Create deep copy of transformation.

        Returns:
            New RigidTransform with cloned data
        """
        if self._axisangle is not None:
            data = self._axisangle.clone()
        elif self._matrix is not None:
            data = self._matrix.clone()
        else:
            raise Exception("Both data are None!")

        return RigidTransform(data, self.trans_first)

    @property
    def device(self) -> DeviceType:
        """Get device where transformation is stored."""
        if self._axisangle is not None:
            return self._axisangle.device
        elif self._matrix is not None:
            return self._matrix.device
        else:
            raise Exception("Both data are None!")

    @property
    def dtype(self) -> torch.dtype:
        """Get data type of transformation parameters."""
        if self._axisangle is not None:
            return self._axisangle.dtype
        elif self._matrix is not None:
            return self._matrix.dtype
        else:
            raise Exception("Both data are None!")

    @staticmethod
    def cat(transforms: Iterable["RigidTransform"]) -> "RigidTransform":
        """
        Concatenate multiple transformations into a batch.

        Args:
            transforms: Iterable of RigidTransform objects

        Returns:
            Single RigidTransform with concatenated data
        """
        matrixs = [t.matrix(trans_first=True) for t in transforms]
        return RigidTransform(torch.cat(matrixs, 0), trans_first=True)

    def __len__(self) -> int:
        """Get batch size (number of transformations)."""
        if self._axisangle is not None:
            return self._axisangle.shape[0]
        elif self._matrix is not None:
            return self._matrix.shape[0]
        else:
            raise Exception("Both data are None!")

    def mean(self, trans_first=True, simple_mean=True) -> "RigidTransform":
        """
        Compute average transformation.

        Args:
            trans_first: Convention for output
            simple_mean: If True, simple average of axis-angles (fast but approximate)
                        If False, proper geodesic mean on SO(3) (slow but accurate)

        Returns:
            Mean transformation

        Note:
            Simple mean works well for small rotations but can produce
            non-unit quaternions for large rotations. Geodesic mean is
            theoretically correct but computationally expensive.
        """
        ax = self.axisangle(trans_first=trans_first)

        if simple_mean:
            # Arithmetic mean (fast)
            ax_mean = ax.mean(0, keepdim=True)
        else:
            # Geodesic mean on rotation manifold (slow)
            meanT = ax[:, 3:].mean(0, keepdim=True)
            meanR = average_rotation(ax[:, :3])
            ax_mean = torch.cat((meanR, meanT), -1)

        return RigidTransform(ax_mean, trans_first=trans_first)


# ============================================================================
# SECTION 3: TRANSFORMATION CONVENTION CONVERSIONS
# ============================================================================


def mat_first2last(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert matrix from 'trans_first=True' to 'trans_first=False'.

    Trans_first=True:  y = R @ (x + t)
    Trans_first=False: y = R @ x + R @ t

    Args:
        mat: Matrix in trans_first=True convention

    Returns:
        Matrix in trans_first=False convention
    """
    R = mat[:, :, :3]
    t = mat[:, :, 3:]
    t = torch.matmul(R, t)  # Transform translation: t_new = R @ t_old
    mat = torch.cat([R, t], -1)
    return mat


def mat_last2first(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert matrix from 'trans_first=False' to 'trans_first=True'.

    This is the inverse of mat_first2last().

    Args:
        mat: Matrix in trans_first=False convention

    Returns:
        Matrix in trans_first=True convention
    """
    R = mat[:, :, :3]
    t = mat[:, :, 3:]
    t = torch.matmul(
        R.transpose(-2, -1), t
    )  # Transform translation: t_new = R^T @ t_old
    mat = torch.cat([R, t], -1)
    return mat


def ax_first2last(axisangle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle from 'trans_first=True' to 'trans_first=False'.

    Args:
        axisangle: Axis-angle in trans_first=True convention

    Returns:
        Axis-angle in trans_first=False convention
    """
    mat = axisangle2mat(axisangle)
    mat = mat_first2last(mat)
    return mat2axisangle(mat)


def ax_last2first(axisangle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle from 'trans_first=False' to 'trans_first=True'.

    Args:
        axisangle: Axis-angle in trans_first=False convention

    Returns:
        Axis-angle in trans_first=True convention
    """
    mat = axisangle2mat(axisangle)
    mat = mat_last2first(mat)
    return mat2axisangle(mat)


def mat_update_resolution(
    mat: torch.Tensor,
    res_from: Union[float, torch.Tensor],
    res_to: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Update transformation matrix for resolution change.

    When resampling images, translations need to be scaled by resolution ratio.
    Rotations are scale-invariant.

    Args:
        mat: Transformation matrix (N, 3, 4)
        res_from: Original resolution
        res_to: New resolution

    Returns:
        Updated matrix with scaled translation

    Example:
        If doubling resolution (0.5mm → 0.25mm), translations double too
    """
    assert mat.dim() == 3
    fac = torch.ones_like(mat[:1, :1])
    fac[..., 3] = res_from / res_to  # Scale translation component
    return mat * fac


def ax_update_resolution(
    ax: torch.Tensor,
    res_from: Union[float, torch.Tensor],
    res_to: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Update axis-angle for resolution change.

    Same as mat_update_resolution() but for axis-angle representation.

    Args:
        ax: Axis-angle (N, 6)
        res_from: Original resolution
        res_to: New resolution

    Returns:
        Updated axis-angle with scaled translation
    """
    assert ax.dim() == 2
    fac = torch.ones_like(ax[:1])
    fac[:, 3:] = res_from / res_to  # Scale translation components
    return ax * fac


# ============================================================================
# SECTION 4: EULER ANGLE CONVERSIONS
# ============================================================================


def mat2euler(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert transformation matrix to Euler angles (XYZ convention, degrees).

    Euler angles represent rotation as three sequential rotations:
    1. Rotate around X axis by RX
    2. Rotate around Y axis by RY
    3. Rotate around Z axis by RZ

    Args:
        mat: Transformation matrix (N, 3, 4)

    Returns:
        Tensor (N, 6) = [TX, TY, TZ, RX, RY, RZ] in mm and degrees

    Note:
        - Suffers from gimbal lock when RY ≈ ±90°
        - Multiple Euler representations can give same rotation
        - Handles gimbal lock case specially (sets RZ=0)

    Warning:
        Euler angles are not ideal for interpolation or optimization.
        Use axis-angle or quaternions for those purposes.
    """
    TOL = 0.000001  # Tolerance for gimbal lock detection

    # Extract translation
    TX = mat[:, 0, 3]
    TY = mat[:, 1, 3]
    TZ = mat[:, 2, 3]

    # Extract Euler angles from rotation matrix
    # Assuming XYZ convention: R = Rz(RZ) @ Ry(RY) @ Rx(RX)
    tmp = torch.asin(-mat[:, 0, 2])  # RY = arcsin(-R[0,2])
    mask = torch.cos(tmp).abs() <= TOL  # Gimbal lock detection

    RX = torch.atan2(mat[:, 1, 2], mat[:, 2, 2])
    RY = tmp
    RZ = torch.atan2(mat[:, 0, 1], mat[:, 0, 0])

    # Handle gimbal lock (when cos(RY) ≈ 0)
    RX[mask] = torch.atan2(-mat[:, 0, 2] * mat[:, 1, 0], -mat[:, 0, 2] * mat[:, 2, 0])[
        mask
    ]
    RZ[mask] = 0  # Arbitrarily set to zero (one degree of freedom lost)

    # Convert radians to degrees
    RX *= 180 / np.pi
    RY *= 180 / np.pi
    RZ *= 180 / np.pi

    return torch.stack((TX, TY, TZ, RX, RY, RZ), -1)


def euler2mat(p: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (degrees) to transformation matrix.

    This is the inverse of mat2euler().

    Args:
        p: Tensor (N, 6) = [TX, TY, TZ, RX, RY, RZ] in mm and degrees

    Returns:
        Transformation matrix (N, 3, 4)

    Note:
        Uses XYZ Euler convention: R = Rz(RZ) @ Ry(RY) @ Rx(RX)
    """
    # Extract components
    tx = p[:, 0]
    ty = p[:, 1]
    tz = p[:, 2]
    rx = p[:, 3]
    ry = p[:, 4]
    rz = p[:, 5]

    # Convert degrees to radians and compute trig functions
    M_PI = np.pi
    cosrx = torch.cos(rx * (M_PI / 180.0))
    cosry = torch.cos(ry * (M_PI / 180.0))
    cosrz = torch.cos(rz * (M_PI / 180.0))
    sinrx = torch.sin(rx * (M_PI / 180.0))
    sinry = torch.sin(ry * (M_PI / 180.0))
    sinrz = torch.sin(rz * (M_PI / 180.0))

    # Initialize with identity
    mat = torch.eye(4, device=p.device)
    mat = mat.reshape((1, 4, 4)).repeat(p.shape[0], 1, 1)

    # Build rotation matrix (XYZ Euler convention)
    mat[:, 0, 0] = cosry * cosrz
    mat[:, 0, 1] = cosry * sinrz
    mat[:, 0, 2] = -sinry
    mat[:, 0, 3] = tx

    mat[:, 1, 0] = sinrx * sinry * cosrz - cosrx * sinrz
    mat[:, 1, 1] = sinrx * sinry * sinrz + cosrx * cosrz
    mat[:, 1, 2] = sinrx * cosry
    mat[:, 1, 3] = ty

    mat[:, 2, 0] = cosrx * sinry * cosrz + sinrx * sinrz
    mat[:, 2, 1] = cosrx * sinry * sinrz - sinrx * cosrz
    mat[:, 2, 2] = cosrx * cosry
    mat[:, 2, 3] = tz

    mat[:, 3, 3] = 1.0

    # Return 3x4 part
    return mat[:, :3, :]


# ============================================================================
# SECTION 5: POINT-BASED TRANSFORMATION DEFINITION
# ============================================================================


def point2mat(p: torch.Tensor) -> torch.Tensor:
    """
    Define transformation from three corresponding points.

    Given three points that define a coordinate frame, compute the
    transformation matrix. Useful for landmark-based registration.

    Args:
        p: Tensor of shape (N, 9) = [p1_x, p1_y, p1_z, p2_x, p2_y, p2_z, p3_x, p3_y, p3_z]
           where:
           - p1: First corner (typically bottom-left)
           - p2: Origin (center point)
           - p3: Second corner (typically bottom-right)

    Returns:
        Transformation matrix (N, 3, 4)

    Algorithm:
        1. v1 = p3 - p1 defines X axis
        2. v2 = p2 - p1
        3. Z axis = v1 × v2 (cross product)
        4. Y axis = Z × v1
        5. Orthonormalize to get rotation matrix R
        6. Translation t places p2 at origin
    """
    p = p.view(-1, 3, 3)
    p1 = p[:, 0]  # First point
    p2 = p[:, 1]  # Origin/center point
    p3 = p[:, 2]  # Third point

    # Define coordinate frame
    v1 = p3 - p1  # Vector along X axis
    v2 = p2 - p1  # Vector in XY plane

    # Construct orthogonal axes
    nz = torch.cross(v1, v2, -1)  # Z axis (perpendicular to plane)
    ny = torch.cross(nz, v1, -1)  # Y axis
    nx = v1  # X axis

    # Build rotation matrix from axes (as columns)
    R = torch.stack((nx, ny, nz), -1)

    # Normalize columns to get orthonormal matrix
    R = R / torch.linalg.norm(R, ord=2, dim=-2, keepdim=True)

    # Compute translation to place p2 at origin
    T = torch.matmul(R.transpose(-2, -1), p2.unsqueeze(-1))

    return torch.cat((R, T), -1)


def mat2point(mat: torch.Tensor, sx: int, sy: int, rs: float) -> torch.Tensor:
    """
    Convert transformation to three points (inverse of point2mat).

    Converts transformation matrix back to landmark representation.
    Useful for visualization or interactive editing.

    Args:
        mat: Transformation matrix (N, 3, 4)
        sx: Image width (in voxels)
        sy: Image height (in voxels)
        rs: Resolution (mm per voxel)

    Returns:
        Tensor (N, 9) of flattened point coordinates

    Note:
        Points define corners and center of image in physical space
    """
    # Define reference points in image space
    p1 = torch.tensor([-(sx - 1) / 2 * rs, -(sy - 1) / 2 * rs, 0]).to(
        dtype=mat.dtype, device=mat.device
    )
    p2 = torch.tensor([0, 0, 0]).to(dtype=mat.dtype, device=mat.device)
    p3 = torch.tensor([(sx - 1) / 2 * rs, -(sy - 1) / 2 * rs, 0]).to(
        dtype=mat.dtype, device=mat.device
    )

    p = torch.stack((p1, p2, p3), 0)
    p = p.unsqueeze(0).unsqueeze(-1)  # Shape: (1, 3, 3, 1)

    # Extract rotation and translation
    R = mat[:, :, :-1].unsqueeze(1)  # (N, 1, 3, 3)
    T = mat[:, :, -1:].unsqueeze(1)  # (N, 1, 3, 1)

    # Transform points: p_world = R @ (p + T)
    p = torch.matmul(R, p + T)

    return p.view(-1, 9)


# ============================================================================
# SECTION 6: APPLYING TRANSFORMATIONS TO POINTS
# ============================================================================


def mat_transform_points(
    mat: torch.Tensor, x: torch.Tensor, trans_first: bool
) -> torch.Tensor:
    """
    Apply transformation matrix to 3D points.

    Args:
        mat: Transformation matrix (*, 3, 4) = [R | t]
        x: Points (*, 3)
        trans_first: Translation convention
            - True: y = R @ (x + t)
            - False: y = R @ x + t

    Returns:
        Transformed points (*, 3)

    Note:
        Supports broadcasting for batch operations
    """
    R = mat[..., :-1]  # (*, 3, 3) rotation
    T = mat[..., -1:]  # (*, 3, 1) translation
    x = x[..., None]  # (*, 3, 1)

    if trans_first:
        x = torch.matmul(R, x + T)
    else:
        x = torch.matmul(R, x) + T

    return x[..., 0]  # Remove last dimension


def ax_transform_points(
    ax: torch.Tensor, x: torch.Tensor, trans_first: bool
) -> torch.Tensor:
    """
    Apply axis-angle transformation to 3D points.

    Converts to matrix internally, then applies transformation.

    Args:
        ax: Axis-angle (*, 6)
        x: Points (*, 3)
        trans_first: Translation convention

    Returns:
        Transformed points (*, 3)
    """
    mat = axisangle2mat(ax.view(-1, 6)).view(ax.shape[:-1] + (3, 4))
    return mat_transform_points(mat, x, trans_first)


def transform_points(transform: RigidTransform, x: torch.Tensor) -> torch.Tensor:
    """
    Apply RigidTransform to 3D points.

    High-level interface for point transformation.

    Args:
        transform: RigidTransform object (batch size N or 1)
        x: Points (N, 3) or (*, 3)

    Returns:
        Transformed points with same shape as x

    Note:
        If transform has batch size 1, it's broadcast to all points
    """
    assert x.ndim == 2 and x.shape[-1] == 3
    trans_first = transform.trans_first
    mat = transform.matrix(trans_first)
    return mat_transform_points(mat, x, trans_first)


# ============================================================================
# SECTION 7: TRANSFORMATION INITIALIZATION
# ============================================================================


def init_stack_transform(
    n_slice: int, gap: float, device: DeviceType
) -> RigidTransform:
    """
    Create default transformation for a stack of slices.

    Places slices evenly spaced along Z axis with identity rotation.
    This is the initial guess before registration.

    Args:
        n_slice: Number of slices
        gap: Spacing between slice centers (mm)
        device: Device to create tensors on

    Returns:
        RigidTransform with n_slice transforms

    Example:
        For 5 slices with 2mm gap:
        - Slice 0: tz = -4mm (two gaps below center)
        - Slice 1: tz = -2mm
        - Slice 2: tz =  0mm (center)
        - Slice 3: tz = +2mm
        - Slice 4: tz = +4mm
    """
    ax = torch.zeros((n_slice, 6), dtype=torch.float32, device=device)

    # Set Z translation for each slice (centered at origin)
    ax[:, -1] = (
        torch.arange(n_slice, dtype=torch.float32, device=device) - (n_slice - 1) / 2.0
    ) * gap

    return RigidTransform(ax, trans_first=True)


def init_zero_transform(n: int, device: DeviceType) -> RigidTransform:
    """
    Create identity transformations (no rotation, no translation).

    Args:
        n: Number of transforms
        device: Device to create tensors on

    Returns:
        RigidTransform representing identity
    """
    return RigidTransform(torch.zeros((n, 6), dtype=torch.float32, device=device))


def average_rotation(R: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic mean of multiple rotations on SO(3).

    This is the "correct" way to average rotations, accounting for
    the non-Euclidean geometry of the rotation group.

    Args:
        R: Axis-angle rotations (N, 3) - only rotation component

    Returns:
        Mean rotation (1, 3) as axis-angle

    Algorithm (Iterative):
        1. Convert axis-angles to quaternions
        2. Resolve quaternion sign ambiguity
        3. Compute initial mean quaternion
        4. Iteratively refine using matrix logarithm/exponential
        5. Convert back to axis-angle

    Note:
        This is expensive (requires scipy) but gives proper geodesic mean.
        For small rotations, simple averaging is often sufficient.

    Reference:
        "On Averaging Rotations" by F. Markley et al.
    """
    import scipy
    from scipy.spatial.transform import Rotation

    dtype = R.dtype
    device = R.device

    # Convert to rotation matrices and quaternions
    Rmat = Rotation.from_rotvec(R.cpu().numpy()).as_matrix()
    R = Rotation.from_rotvec(R.cpu().numpy()).as_quat()

    # Resolve quaternion sign ambiguity (q and -q represent same rotation)
    # Align all quaternions to have positive dot product with first one
    for i in range(R.shape[0]):
        if np.linalg.norm(R[i] + R[0]) < np.linalg.norm(R[i] - R[0]):
            R[i] *= -1

    # Initial mean estimate (simple average)
    barR = np.mean(R, 0)
    barR = barR / np.linalg.norm(barR)  # Normalize

    # Iterative refinement using Lie group structure
    S_new = S = Rotation.from_quat(barR).as_matrix()
    R = Rmat
    i = 0

    while np.all(np.isreal(S_new)) and np.all(np.isfinite(S_new)) and i < 10:
        S = S_new
        i += 1

        sum_vmatrix_normed = np.zeros((3, 3))
        sum_inv_norm_vmatrix = 0

        # Compute weighted update direction
        for j in range(R.shape[0]):
            # Log map: rotation difference in tangent space
            vmatrix = scipy.linalg.logm(np.matmul(R[j], np.linalg.inv(S)))
            vmatrix_normed = vmatrix / np.linalg.norm(vmatrix, ord=2, axis=(0, 1))
            sum_vmatrix_normed += vmatrix_normed
            sum_inv_norm_vmatrix += 1 / np.linalg.norm(vmatrix, ord=2, axis=(0, 1))

        # Compute update step
        delta = sum_vmatrix_normed / sum_inv_norm_vmatrix

        if np.all(np.isfinite(delta)):
            # Exp map: move in tangent space direction
            S_new = np.matmul(scipy.linalg.expm(delta), S)
        else:
            break

    # Convert back to axis-angle
    S = Rotation.from_matrix(S).as_rotvec()
    return torch.tensor(S[None], dtype=dtype, device=device)


# ============================================================================
# SECTION 8: NIFTI COORDINATE SYSTEM CONVERSIONS
# ============================================================================


def compare_resolution_affine(r1, a1, r2, a2, s1, s2) -> bool:
    """
    Check if two images have compatible resolution and affine matrices.

    Used to verify that an image and its mask are compatible before
    combining them.

    Args:
        r1: Resolution array for image 1
        a1: Affine matrix for image 1
        r2: Resolution array for image 2
        a2: Affine matrix for image 2
        s1: Shape tuple for image 1
        s2: Shape tuple for image 2

    Returns:
        True if images match (within tolerance), False otherwise

    Note:
        Uses 1e-3 tolerance for floating point comparisons
    """
    r1 = np.array(r1)
    a1 = np.array(a1)
    r2 = np.array(r2)
    a2 = np.array(a2)

    # Check shapes match
    if s1 != s2:
        return False

    # Check resolution arrays match
    if r1.shape != r2.shape:
        return False
    if np.amax(np.abs(r1 - r2)) > 1e-3:
        return False

    # Check affine matrices match
    if a1.shape != a2.shape:
        return False
    if np.amax(np.abs(a1 - a2)) > 1e-3:
        return False

    return True


def affine2transformation(
    volume: torch.Tensor,
    mask: torch.Tensor,
    resolutions: np.ndarray,
    affine: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, RigidTransform]:
    """
    Convert NIfTI affine matrix to internal rigid transformation format.

    NIfTI stores a 4x4 affine matrix that maps voxel indices to physical
    coordinates (in mm). This function:
    1. Extracts rotation and translation from affine
    2. Normalizes by voxel resolution
    3. Creates per-slice transformations
    4. Handles coordinate system flips (negative determinant)

    Args:
        volume: Image data (D, H, W)
        mask: Binary mask (D, H, W)
        resolutions: Voxel spacing [rx, ry, rz] in mm
        affine: 4x4 NIfTI affine matrix

    Returns:
        volume: Potentially flipped volume
        mask: Potentially flipped mask
        transformation: RigidTransform with per-slice transforms (length D)

    Note:
        - Internal convention: (D, H, W) = (z, y, x)
        - NIfTI convention: affine maps (x, y, z) indices to (x, y, z) mm
        - Handles oblique acquisitions and arbitrary orientations
    """
    device = volume.device
    d, h, w = volume.shape

    # Extract 3x3 rotation from affine
    R = affine[:3, :3]

    # Check if coordinate system is left-handed (negative determinant)
    negative_det = np.linalg.det(R) < 0

    # Extract translation
    T = affine[:3, -1:]

    # Normalize rotation by resolution (remove scaling component)
    R = R @ np.linalg.inv(np.diag(resolutions))

    # Adjust for coordinate system origin at volume center
    T0 = np.array([(w - 1) / 2 * resolutions[0], (h - 1) / 2 * resolutions[1], 0])
    T = np.linalg.inv(R) @ T + T0.reshape(3, 1)

    # Create per-slice translations (evenly spaced along z)
    tz = (
        torch.arange(0, d, device=device, dtype=torch.float32) * resolutions[2]
        + T[2].item()
    )
    tx = torch.ones_like(tz) * T[0].item()
    ty = torch.ones_like(tz) * T[1].item()
    t = torch.stack((tx, ty, tz), -1).view(-1, 3, 1)  # (D, 3, 1)

    # Replicate rotation for all slices
    R = torch.tensor(R, device=device).unsqueeze(0).repeat(d, 1, 1)  # (D, 3, 3)

    # Handle negative determinant by flipping along x-axis
    if negative_det:
        volume = torch.flip(volume, (-1,))  # Flip width
        mask = torch.flip(mask, (-1,))
        t[:, 0, -1] *= -1  # Negate x translation
        R[:, :, 0] *= -1  # Negate first column (x axis)

    # Create transformation object
    transformation = RigidTransform(
        torch.cat((R, t), -1).to(torch.float32),  # (D, 3, 4)
        trans_first=True,
    )

    return volume, mask, transformation


def transformation2affine(
    volume: torch.Tensor,
    transformation: RigidTransform,
    resolution_x: float,
    resolution_y: float,
    resolution_z: float,
) -> np.ndarray:
    """
    Convert internal rigid transformation to NIfTI affine matrix.

    This is the inverse of affine2transformation(), used when saving volumes.

    Args:
        volume: Image data (D, H, W)
        transformation: RigidTransform (assumes single transform, not per-slice)
        resolution_x: Voxel spacing in x (mm)
        resolution_y: Voxel spacing in y (mm)
        resolution_z: Voxel spacing in z (mm)

    Returns:
        affine: 4x4 NIfTI affine matrix

    Note:
        Assumes transformation represents the average/canonical orientation
        of the volume, not per-slice transforms.
    """
    # Get transformation matrix
    mat = transformation.matrix(trans_first=True).detach().cpu().numpy()
    assert mat.shape[0] == 1, "Expected single transformation"

    R = mat[0, :, :-1]  # 3x3 rotation
    T = mat[0, :, -1:]  # 3x1 translation

    d, h, w = volume.shape
    affine = np.eye(4)

    # Adjust translation from center-origin to corner-origin
    T[0] -= (w - 1) / 2 * resolution_x
    T[1] -= (h - 1) / 2 * resolution_y
    T[2] -= (d - 1) / 2 * resolution_z

    # Apply rotation to translation
    T = R @ T.reshape(3, 1)

    # Scale rotation by voxel resolution
    R = R @ np.diag([resolution_x, resolution_y, resolution_z])

    # Build 4x4 affine
    affine[:3, :] = np.concatenate((R, T), -1)

    return affine


# ============================================================================
# SECTION 9: NIFTI FILE I/O
# ============================================================================


def save_nii_volume(
    path: PathType,
    volume: Union[torch.Tensor, np.ndarray],
    affine: Optional[Union[torch.Tensor, np.ndarray]],
) -> None:
    """
    Save 3D volume to NIfTI file (.nii or .nii.gz).

    Handles coordinate system conversion and metadata.

    Args:
        path: Output file path
        volume: 3D or 4D array/tensor
            - (D, H, W): 3D volume
            - (N, 1, H, W): 4D with singleton channel
        affine: 4x4 affine matrix (if None, uses identity)

    Note:
        - Transposes from internal (D,H,W) to NIfTI (W,H,D) convention
        - Sets proper units (mm) and coordinate codes
        - Converts bool to int16 (NIfTI doesn't support bool)
    """
    # Handle 4D input with singleton channel
    assert len(volume.shape) == 3 or (len(volume.shape) == 4 and volume.shape[1] == 1)
    if len(volume.shape) == 4:
        volume = volume.squeeze(1)

    # Convert to numpy and transpose to NIfTI convention
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy().transpose(2, 1, 0)
    else:
        volume = volume.transpose(2, 1, 0)

    # Handle affine
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    if affine is None:
        affine = np.eye(4)

    # NIfTI doesn't support bool dtype
    if volume.dtype == bool and isinstance(volume, np.ndarray):
        volume = volume.astype(np.int16)

    # Create NIfTI image
    img = nib.nifti1.Nifti1Image(volume, affine)

    # Set metadata
    img.header.set_xyzt_units(2)  # Spatial units: mm
    img.header.set_qform(affine, code="aligned")  # Aligned coordinates
    img.header.set_sform(affine, code="scanner")  # Scanner coordinates

    # Save to disk
    nib.save(img, os.fspath(path))


def load_nii_volume(path: PathType) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load 3D volume from NIfTI file.

    Args:
        path: Path to .nii or .nii.gz file

    Returns:
        volume: 3D numpy array (D, H, W), dtype float32
        resolutions: Array [rx, ry, rz] with voxel spacing (mm)
        affine: 4x4 affine transformation matrix

    Note:
        - Handles 4D files with singleton extra dimensions
        - Transposes from NIfTI (W,H,D) to internal (D,H,W)
        - Falls back to qform if sform contains NaNs
    """
    img = nib.load(os.fspath(path))

    # Check dimensionality
    dim = img.header["dim"]
    assert dim[0] == 3 or (dim[0] > 3 and all(d == 1 for d in dim[4:])), (
        f"Expect a 3D volume but the input is {dim[0]}D"
    )

    # Load and convert to float32
    volume = img.get_fdata().astype(np.float32)

    # Squeeze extra singleton dimensions
    while volume.ndim > 3:
        volume = volume.squeeze(-1)

    # Transpose from NIfTI (W, H, D) to internal (D, H, W)
    volume = volume.transpose(2, 1, 0)

    # Extract voxel resolution from header
    resolutions = img.header["pixdim"][1:4]

    # Get affine matrix (prefer sform, fallback to qform)
    affine = img.affine
    if np.any(np.isnan(affine)):
        affine = img.get_qform()

    return volume, resolutions, affine


# ============================================================================
# SECTION 10: IMAGE DATA STRUCTURES
# ============================================================================
# Classes for representing medical images: Image, Slice, Volume, Stack


class _Data(object):
    """
    Base class for all image data structures.

    All medical imaging data consists of:
    - data: The actual image intensities (MRI signal)
    - mask: Boolean array indicating valid regions
    - transformation: Rigid transform to world coordinates

    This base class provides:
    - Property validation (type, shape, device checks)
    - Common attributes and methods
    - Cloning functionality
    """

    def __init__(
        self,
        data: torch.Tensor,
        mask: Optional[torch.Tensor],
        transformation: Optional[RigidTransform],
    ) -> None:
        """
        Initialize base data structure.

        Args:
            data: Image data tensor
            mask: Boolean mask (if None, creates all-ones)
            transformation: Rigid transform (if None, creates identity)
        """
        if mask is None:
            mask = torch.ones_like(data, dtype=torch.bool)
        if transformation is None:
            transformation = init_zero_transform(1, data.device)

        self.data = data
        self.mask = mask
        self.transformation = transformation

    def check_data(self, value) -> None:
        """Validate that data is a torch.Tensor."""
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Data must be Tensor!")

    def check_mask(self, value) -> None:
        """Validate mask properties."""
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Mask must be Tensor!")
        if value.shape != self.shape:
            raise RuntimeError("Mask has a shape different from image!")
        if value.dtype != torch.bool:
            raise RuntimeError("Mask must be bool!")
        if value.device != self.device:
            raise RuntimeError("The device of mask is different!")

    def check_transformation(self, value) -> None:
        """Validate transformation properties."""
        if not isinstance(value, RigidTransform):
            raise RuntimeError("Transformation must be RigidTransform")
        if value.device != self.device:
            raise RuntimeError("The device of transformation must be the same as data!")

    @property
    def data(self) -> torch.Tensor:
        """Image data tensor."""
        return self._data

    @data.setter
    def data(self, value: torch.Tensor) -> None:
        self.check_data(value)
        self._data = value

    @property
    def mask(self) -> torch.Tensor:
        """Boolean mask indicating valid data regions."""
        return self._mask

    @mask.setter
    def mask(self, value: torch.Tensor) -> None:
        self.check_mask(value)
        self._mask = value

    @property
    def transformation(self) -> RigidTransform:
        """Rigid transformation to world coordinates."""
        return self._transformation

    @transformation.setter
    def transformation(self, value: RigidTransform) -> None:
        self.check_transformation(value)
        self._transformation = value

    @property
    def device(self) -> DeviceType:
        """Device where tensors are stored."""
        return self.data.device

    @property
    def shape(self) -> torch.Size:
        """Shape of the data tensor."""
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        """Data type of image tensor."""
        return self.data.dtype

    def clone(self, *, zero: bool = False, deep: bool = True) -> "_Data":
        """Create a copy of this data structure (to be implemented by subclasses)."""
        raise NotImplementedError()

    def _clone_dict(self, zero: bool = False, deep: bool = True) -> Dict:
        """
        Helper for cloning - creates dictionary of cloned attributes.

        Args:
            zero: If True, zero out data and mask
            deep: If True, deep copy; otherwise shallow copy

        Returns:
            Dictionary with cloned attributes
        """
        data = self.data
        mask = self.mask
        transformation = self.transformation

        if zero:
            data = torch.zeros_like(data)
            mask = torch.zeros_like(mask)
        elif deep:
            data = data.clone()
            mask = mask.clone()

        if deep:
            transformation = transformation.clone()

        return {
            "data": data,
            "mask": mask,
            "transformation": self.transformation,
        }


class Image(_Data):
    """
    Base class for 3D medical images with resolution information.

    Extends _Data with:
    - resolution_x, resolution_y, resolution_z: Physical voxel spacing (mm)
    - Methods for saving, rescaling, coordinate transformations

    This is the parent class for both Volume (true 3D) and Slice (2D with depth=1).
    """

    def __init__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        transformation: Optional[RigidTransform] = None,
        resolution_x: Union[float, torch.Tensor] = 1.0,
        resolution_y: Union[float, torch.Tensor, None] = None,
        resolution_z: Union[float, torch.Tensor, None] = None,
    ) -> None:
        """
        Initialize image with resolution.

        Args:
            image: 3D tensor (D, H, W)
            mask: Boolean mask (optional)
            transformation: Rigid transform (optional)
            resolution_x: Voxel spacing in x (width) mm
            resolution_y: Voxel spacing in y (height) mm (defaults to resolution_x)
            resolution_z: Voxel spacing in z (depth) mm (defaults to resolution_x)
        """
        super().__init__(image, mask, transformation)

        # Default to isotropic if not specified
        if resolution_y is None:
            resolution_y = resolution_x
        if resolution_z is None:
            resolution_z = resolution_x

        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z

    def check_data(self, value) -> None:
        """Ensure image is 3D."""
        super().check_data(value)
        if value.ndim != 3:
            raise RuntimeError("The dimension of image must be 3!")

    def check_transformation(self, value) -> None:
        """Ensure transformation has length 1 (single transform)."""
        super().check_transformation(value)
        if len(value) != 1:
            raise RuntimeError("The len of transformation must be 1!")

    @property
    def image(self) -> torch.Tensor:
        """Alias for data property."""
        return self.data

    @image.setter
    def image(self, value: torch.Tensor) -> None:
        self.data = value

    def _clone_dict(self, zero: bool = False, deep: bool = True) -> Dict:
        """Extend parent clone_dict with resolution info."""
        d = super()._clone_dict(zero, deep)
        d["resolution_x"] = float(self.resolution_x)
        d["resolution_y"] = float(self.resolution_y)
        d["resolution_z"] = float(self.resolution_z)
        d["image"] = d.pop("data")
        return d

    @property
    def shape_xyz(self) -> torch.Tensor:
        """Get shape in XYZ order (width, height, depth)."""
        return torch.tensor(self.image.shape[::-1], device=self.image.device)

    @property
    def resolution_xyz(self) -> torch.Tensor:
        """Get resolution as tensor [rx, ry, rz]."""
        return torch.tensor(
            [self.resolution_x, self.resolution_y, self.resolution_z],
            device=self.image.device,
        )

    def save(self, path: PathType, masked: bool = True) -> None:
        """
        Save image to NIfTI file.

        Args:
            path: Output path (.nii or .nii.gz)
            masked: If True, multiply by mask before saving
        """
        affine = transformation2affine(
            self.image,
            self.transformation,
            float(self.resolution_x),
            float(self.resolution_y),
            float(self.resolution_z),
        )

        if masked:
            output_volume = self.image * self.mask.to(self.image.dtype)
        else:
            output_volume = self.image

        save_nii_volume(path, output_volume, affine)

    def save_mask(self, path: PathType) -> None:
        """Save mask to NIfTI file."""
        affine = transformation2affine(
            self.image,
            self.transformation,
            float(self.resolution_x),
            float(self.resolution_y),
            float(self.resolution_z),
        )
        output_volume = self.mask.to(self.image.dtype)
        save_nii_volume(path, output_volume, affine)

    @property
    def xyz_masked(self) -> torch.Tensor:
        """
        Get physical coordinates (mm) of all masked voxels in world space.

        Returns:
            Tensor (N, 3) where N is number of masked voxels
        """
        return transform_points(self.transformation, self.xyz_masked_untransformed)

    @property
    def xyz_masked_untransformed(self) -> torch.Tensor:
        """
        Get physical coordinates of masked voxels in image space.

        Converts voxel indices to millimeter coordinates:
        - Centers coordinate system at volume center
        - Scales by resolution

        Returns:
            Tensor (N, 3) where N is number of masked voxels
        """
        # Get indices of masked voxels, flip to (i, j, k) order
        kji = torch.flip(torch.nonzero(self.mask), (-1,))

        # Center and scale
        return (kji - (self.shape_xyz - 1) / 2) * self.resolution_xyz

    @property
    def v_masked(self) -> torch.Tensor:
        """
        Get intensity values of all masked voxels.

        Returns:
            1D tensor (N,) where N is number of masked voxels
        """
        return self.image[self.mask]

    def rescale(
        self, intensity_mean: Union[float, torch.Tensor], masked: bool = True
    ) -> None:
        """
        Rescale image intensities to target mean (in-place).

        Used for intensity normalization across images.

        Args:
            intensity_mean: Target mean intensity
            masked: If True, compute mean only over masked region
        """
        if masked:
            scale_factor = intensity_mean / self.image[self.mask].mean()
        else:
            scale_factor = intensity_mean / self.image.mean()

        self.image *= scale_factor

    @staticmethod
    def like(
        old: "Image",
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deep: bool = True,
    ) -> "Image":
        """
        Create new Image with same properties as old, optionally with new data.

        Args:
            old: Template image
            image: New image data (if None, copies from old)
            mask: New mask (if None, copies from old)
            deep: If True, deep copy; otherwise shallow

        Returns:
            New Image instance
        """
        if image is None:
            image = old.image.clone() if deep else old.image
        if mask is None:
            mask = old.mask.clone() if deep else old.mask

        transformation = old.transformation.clone() if deep else old.transformation

        return old.__class__(
            image=image,
            mask=mask,
            transformation=transformation,
            resolution_x=old.resolution_x,
            resolution_y=old.resolution_y,
            resolution_z=old.resolution_z,
        )


class Slice(Image):
    """
    Single 2D slice from an MRI scan.

    Shape: (1, H, W) - essentially 3D image with depth=1.

    Used for slice-to-volume reconstruction where we have multiple
    2D acquisitions that need to be combined into a 3D volume.
    """

    def check_data(self, value) -> None:
        """Ensure first dimension is 1 (single slice)."""
        super().check_data(value)
        if value.shape[0] != 1:
            raise RuntimeError("The shape of a slice must be (1, H, W)!")

    def clone(self, *, zero: bool = False, deep: bool = True) -> Slice:
        """Create a copy of this slice."""
        return Slice(**self._clone_dict(zero, deep))

    def resample(
        self,
        resolution_new: Union[float, Sequence],
    ) -> Slice:
        """
        Resample slice to new resolution using interpolation.

        Args:
            resolution_new: Target resolution
                - Single float: isotropic in-plane
                - [rx, ry]: anisotropic in-plane
                - [rx, ry, rz]: includes through-plane

        Returns:
            New Slice at different resolution
        """
        # Handle different input formats
        if isinstance(resolution_new, float) or len(resolution_new) == 1:
            resolution_new = [resolution_new, resolution_new]

        if len(resolution_new) == 3:
            resolution_z_new = resolution_new[-1]
            resolution_new = resolution_new[:-1]
        else:
            resolution_z_new = self.resolution_z

        # Resample image
        image = resample(
            self.image[None],
            (self.resolution_x, self.resolution_y),
            resolution_new,
        )[0]

        # Resample mask (threshold to keep binary)
        mask = (
            resample(
                self.mask[None].float(),
                (self.resolution_x, self.resolution_y),
                resolution_new,
            )[0]
            > 0
        )

        # Create new slice
        new_slice = cast(Slice, Slice.like(self, image, mask, deep=True))
        new_slice.resolution_z = resolution_z_new

        return new_slice


class Volume(Image):
    """
    3D medical imaging volume.

    Can represent:
    - Reconstructed high-resolution volume
    - Mask volume
    - Template/atlas image

    Provides methods for:
    - Sampling at arbitrary 3D coordinates
    - Resampling to different resolutions/orientations
    """

    def sample_points(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Sample volume intensities at arbitrary 3D coordinates.

        Uses trilinear interpolation for non-grid locations.

        Args:
            xyz: Coordinates in world space (..., 3)

        Returns:
            Interpolated values (...)
        """
        shape = xyz.shape[:-1]

        # Transform to image space
        xyz = transform_points(self.transformation.inv(), xyz.view(-1, 3))

        # Normalize to [-1, 1] for grid_sample
        xyz = xyz / ((self.shape_xyz - 1) * self.resolution_xyz / 2)

        # Sample using trilinear interpolation
        return F.grid_sample(
            self.image[None, None],
            xyz.view(1, 1, 1, -1, 3),
            align_corners=True,
        ).view(shape)

    def resample(
        self,
        resolution_new: Optional[Union[float, torch.Tensor]],
        transformation_new: Optional[RigidTransform],
    ) -> Volume:
        """
        Resample volume to new resolution and/or orientation.

        Creates new volume with:
        - Different voxel spacing
        - Different orientation
        - Auto-computed size to fit all data

        Args:
            resolution_new: Target resolution (isotropic or [rx, ry, rz])
            transformation_new: Target orientation (if None, keeps current)

        Returns:
            New Volume at target resolution/orientation
        """
        if transformation_new is None:
            transformation_new = self.transformation

        R = transformation_new.matrix()[0, :3, :3]
        dtype = R.dtype
        device = R.device

        # Handle resolution formats
        if resolution_new is None:
            resolution_new = self.resolution_xyz
        elif isinstance(resolution_new, float) or resolution_new.numel == 1:
            resolution_new = torch.tensor(
                [resolution_new] * 3, dtype=dtype, device=device
            )

        # Get masked voxel coordinates
        xyz = self.xyz_masked

        # Rotate to new orientation
        xyz = torch.matmul(torch.inverse(R), xyz.view(-1, 3, 1))[..., 0]

        # Compute bounding box with padding
        xyz_min = xyz.amin(0) - resolution_new * 10
        xyz_max = xyz.amax(0) + resolution_new * 10
        shape_xyz = ((xyz_max - xyz_min) / resolution_new).ceil().long()

        # Create transformation for new volume
        mat = torch.zeros((1, 3, 4), dtype=R.dtype, device=R.device)
        mat[0, :, :3] = R
        mat[0, :, -1] = xyz_min + (shape_xyz - 1) / 2 * resolution_new

        # Create sampling grid
        xyz = meshgrid(shape_xyz, resolution_new, xyz_min, device, True)
        xyz = torch.matmul(R, xyz[..., None])[..., 0]

        # Sample at grid points
        v = self.sample_points(xyz)

        return Volume(
            v,
            v > 0,
            RigidTransform(mat, trans_first=True),
            resolution_new[0].item(),
            resolution_new[1].item(),
            resolution_new[2].item(),
        )

    def clone(self, *, zero: bool = False, deep: bool = True) -> Volume:
        """Create a copy of this volume."""
        return Volume(**self._clone_dict(zero))

    @staticmethod
    def zeros(
        shape: Tuple,
        resolution_x: float,
        resolution_y: Optional[float] = None,
        resolution_z: Optional[float] = None,
        device: DeviceType = None,
    ) -> Volume:
        """
        Create zero-filled volume.

        Useful for initializing reconstruction targets.

        Args:
            shape: Volume shape (D, H, W)
            resolution_x: Voxel spacing in x
            resolution_y: Voxel spacing in y (defaults to resolution_x)
            resolution_z: Voxel spacing in z (defaults to resolution_x)
            device: Device to create on

        Returns:
            New Volume filled with zeros
        """
        image = torch.zeros(shape, dtype=torch.float32, device=device)
        mask = torch.ones_like(image, dtype=torch.bool)

        return Volume(
            image,
            mask,
            transformation=None,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            resolution_z=resolution_z,
        )


class Stack(_Data):
    """
    Stack of 2D slices from a single MRI acquisition.

    Shape: (N, 1, H, W) where:
    - N: Number of slices
    - 1: Channel dimension (always 1 for MRI)
    - H, W: Height and width of each slice

    Attributes:
        resolution_x, resolution_y: In-plane voxel spacing (mm)
        thickness: Physical thickness of each slice (mm)
        gap: Spacing between slice centers (mm)
        transformation: Per-slice rigid transforms (length N)
        name: Optional identifier

    Typical MRI might have:
    - In-plane: 0.5mm × 0.5mm
    - Thickness: 2mm
    - Gap: 2mm (or more with spacing)
    """

    def __init__(
        self,
        slices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        transformation: Optional[RigidTransform] = None,
        resolution_x: float = 1.0,
        resolution_y: Optional[float] = None,
        thickness: Optional[float] = None,
        gap: Optional[float] = None,
        name: str = "",
    ) -> None:
        """
        Initialize stack of slices.

        Args:
            slices: 4D tensor (N, 1, H, W)
            mask: Boolean mask (optional)
            transformation: Per-slice transforms (if None, creates default spacing)
            resolution_x: In-plane resolution in x
            resolution_y: In-plane resolution in y (defaults to resolution_x)
            thickness: Slice thickness (defaults to gap or resolution_x)
            gap: Inter-slice spacing (defaults to thickness)
            name: Optional identifier
        """
        if resolution_y is None:
            resolution_y = resolution_x
        if thickness is None:
            thickness = gap if gap is not None else resolution_x
        if gap is None:
            gap = thickness
        if transformation is None:
            transformation = init_stack_transform(slices.shape[0], gap, slices.device)

        super().__init__(slices, mask, transformation)

        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.thickness = thickness
        self.gap = gap
        self.name = name

    def check_data(self, value) -> None:
        """Ensure data is 4D with shape (N, 1, H, W)."""
        super().check_data(value)
        if value.ndim != 4:
            raise RuntimeError("Stack must be 4D data")
        if value.shape[1] != 1:
            raise RuntimeError("Stack must has shape (N, 1, H, W)")

    def check_transformation(self, value) -> None:
        """Ensure one transformation per slice."""
        super().check_transformation(value)
        if len(value) != self.slices.shape[0]:
            raise RuntimeError(
                "The number of transformatons is not equal to the number of slices!"
            )

    @property
    def slices(self) -> torch.Tensor:
        """Alias for data property."""
        return self.data

    @slices.setter
    def slices(self, value: torch.Tensor) -> None:
        self.data = value

    def __len__(self) -> int:
        """Number of slices."""
        return self.slices.shape[0]

    def __getitem__(self, idx) -> Union[Slice, List[Slice]]:
        """
        Index into stack to get slice(s).

        Args:
            idx: Integer, slice, or list

        Returns:
            Single Slice (if idx is int) or list of Slices
        """
        slices = self.slices[idx]
        masks = self.mask[idx]
        transformation = self.transformation[idx]

        # Single slice
        if slices.ndim < self.slices.ndim:
            return Slice(
                slices,
                masks,
                transformation,
                self.resolution_x,
                self.resolution_y,
                self.thickness,
            )
        # Multiple slices
        else:
            return [
                Slice(
                    slices[i],
                    masks[i],
                    transformation[i],
                    self.resolution_x,
                    self.resolution_y,
                    self.thickness,
                )
                for i in range(len(transformation))
            ]

    def get_substack(
        self, idx_from: Optional[int] = None, idx_to: Optional[int] = None
    ) -> Stack:
        """
        Extract subset of slices as new Stack.

        Args:
            idx_from: Start index (if idx_to None, gets single slice)
            idx_to: End index (exclusive)

        Returns:
            New Stack with subset
        """
        if idx_to is None:
            slices = self.slices[idx_from]
            masks = self.mask[idx_from]
            transformation = self.transformation[idx_from]
        else:
            slices = self.slices[idx_from:idx_to]
            masks = self.mask[idx_from:idx_to]
            transformation = self.transformation[idx_from:idx_to]

        return Stack(
            slices,
            masks,
            transformation,
            self.resolution_x,
            self.resolution_y,
            self.thickness,
            self.gap,
            self.name,
        )

    def get_mask_volume(self) -> Volume:
        """
        Convert stack to Volume representation of mask.

        Uses mean transformation across slices.

        Returns:
            Volume with mask as float data
        """
        mask = self.mask.squeeze(1).clone()

        return Volume(
            image=mask.float(),
            mask=mask > 0,
            transformation=self.transformation.mean(),
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
            resolution_z=self.gap,
        )

    def get_volume(self, copy: bool = True) -> Volume:
        """
        Convert stack to Volume representation.

        Treats stack as 3D volume:
        - Removes channel dimension
        - Uses average transformation
        - Uses gap as z-resolution

        Args:
            copy: If True, copies data; otherwise shares memory

        Returns:
            Volume representation
        """
        image = self.slices.squeeze(1)
        mask = self.mask.squeeze(1)

        if copy:
            image = image.clone()
            mask = mask.clone()

        return Volume(
            image=image,
            mask=mask,
            transformation=self.transformation.mean(),
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
            resolution_z=self.gap,
        )

    def apply_volume_mask(self, mask: Volume) -> None:
        """
        Apply 3D volume mask to all slices.

        For each slice, samples volume mask at slice coordinates
        and updates slice mask.

        Args:
            mask: Volume mask to apply
        """
        for i in range(len(self)):
            s = self[i]
            assign_mask = self.mask[i].clone()
            self.mask[i][assign_mask] = mask.sample_points(s.xyz_masked) > 0

    def _clone_dict(self, zero: bool = False, deep: bool = True) -> Dict:
        """Extend parent clone_dict with stack attributes."""
        d = super()._clone_dict(zero, deep)
        d["slices"] = d.pop("data")
        d["resolution_x"] = float(self.resolution_x)
        d["resolution_y"] = float(self.resolution_y)
        d["thickness"] = float(self.thickness)
        d["gap"] = float(self.gap)
        d["name"] = self.name
        return d

    def clone(self, *, zero: bool = False, deep: bool = True) -> Stack:
        """Create copy of this stack."""
        return Stack(**self._clone_dict(zero, deep))

    @staticmethod
    def like(
        stack: Stack,
        slices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deep: bool = True,
    ) -> Stack:
        """
        Create new Stack with same properties, optionally with new data.

        Args:
            stack: Template stack
            slices: New slice data (if None, copies from stack)
            mask: New mask (if None, copies from stack)
            deep: If True, deep copy; otherwise shallow

        Returns:
            New Stack instance
        """
        if slices is None:
            slices = stack.slices.clone() if deep else stack.slices
        if mask is None:
            mask = stack.mask.clone() if deep else stack.mask

        transformation = stack.transformation.clone() if deep else stack.transformation

        return Stack(
            slices=slices,
            mask=mask,
            transformation=transformation,
            resolution_x=stack.resolution_x,
            resolution_y=stack.resolution_y,
            thickness=stack.thickness,
            gap=stack.gap,
        )

    @staticmethod
    def pad_stacks(stacks: List[Stack]) -> List[Stack]:
        """
        Pad all stacks to same spatial dimensions.

        Finds largest height/width and zero-pads smaller stacks.

        Args:
            stacks: List of Stack objects

        Returns:
            List of padded Stack objects
        """
        # Find maximum size
        size_max = max([max(s.shape[-2:]) for s in stacks])

        lists_pad = []
        for s in stacks:
            # Check if padding needed
            if s.shape[-1] < size_max or s.shape[-2] < size_max:
                # Compute padding
                dx1 = (size_max - s.shape[-1]) // 2
                dx2 = (size_max - s.shape[-1]) - dx1
                dy1 = (size_max - s.shape[-2]) // 2
                dy2 = (size_max - s.shape[-2]) - dy1

                # Pad
                data = F.pad(s.data, (dx1, dx2, dy1, dy2))
                mask = F.pad(s.mask, (dx1, dx2, dy1, dy2))
            else:
                data = s.data
                mask = s.mask

            lists_pad.append(s.__class__.like(s, data, mask, deep=False))

        return lists_pad

    @staticmethod
    def cat(inputs: List[Union[Slice, Stack]]) -> Stack:
        """
        Concatenate multiple Slices/Stacks into single Stack.

        Resolution taken from first input.

        Args:
            inputs: List of Slice and/or Stack objects

        Returns:
            Combined Stack
        """
        data = []
        mask = []
        transformation = []

        for i, inp in enumerate(inputs):
            if isinstance(inp, Slice):
                data.append(inp.image[None])
                mask.append(inp.mask[None])
                transformation.append(inp.transformation)

                if i == 0:
                    resolution_x = float(inp.resolution_x)
                    resolution_y = float(inp.resolution_y)
                    thickness = float(inp.resolution_z)
                    gap = float(inp.resolution_z)

            elif isinstance(inp, Stack):
                data.append(inp.slices)
                mask.append(inp.mask)
                transformation.append(inp.transformation)

                if i == 0:
                    resolution_x = inp.resolution_x
                    resolution_y = inp.resolution_y
                    thickness = inp.thickness
                    gap = inp.gap
            else:
                raise TypeError("unknown type!")

        return Stack(
            slices=torch.cat(data, 0),
            mask=torch.cat(mask, 0),
            transformation=RigidTransform.cat(transformation),
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            thickness=thickness,
            gap=gap,
        )

    def init_stack_transform(self) -> RigidTransform:
        """
        Create default transformation for this stack.

        Places slices evenly along z-axis with identity rotation.

        Returns:
            RigidTransform with per-slice transforms
        """
        return init_stack_transform(len(self), self.gap, self.device)


# ============================================================================
# SECTION 11: HIGH-LEVEL FILE I/O
# ============================================================================

MASK_PREFIX = "mask_"


def save_slices(folder: PathType, images: List[Slice], sep: bool = False) -> None:
    """
    Save list of slices to folder.

    Args:
        folder: Output directory
        images: List of Slice objects
        sep: If True, saves images and masks separately
    """
    for i, image in enumerate(images):
        if sep:
            image.save(os.path.join(folder, f"{i}.nii.gz"), masked=False)
            image.save_mask(os.path.join(folder, f"{MASK_PREFIX}{i}.nii.gz"))
        else:
            image.save(os.path.join(folder, f"{i}.nii.gz"), masked=True)


def load_slices(
    folder: PathType, device: DeviceType = torch.device("cpu")
) -> List[Slice]:
    """
    Load slices from folder of NIfTI files.

    Looks for: 0.nii.gz, 1.nii.gz, ...
    Optionally loads: mask_0.nii.gz, mask_1.nii.gz, ...

    Args:
        folder: Directory with slice files
        device: Device to load onto

    Returns:
        List of Slice objects, sorted by index
    """
    slices = []
    ids = []

    for f in os.listdir(folder):
        # Skip non-NIfTI
        if not (f.endswith("nii") or f.endswith("nii.gz")):
            continue

        # Skip mask files
        if f.startswith(MASK_PREFIX):
            continue

        # Extract index
        ids.append(int(f.split(".nii")[0]))

        # Load image
        slice_data, resolutions, affine = load_nii_volume(os.path.join(folder, f))
        slice_tensor = torch.tensor(slice_data, device=device)

        # Load mask if exists
        mask_path = os.path.join(folder, MASK_PREFIX + f)
        if os.path.exists(mask_path):
            mask_data, _, _ = load_nii_volume(mask_path)
            mask_tensor = torch.tensor(mask_data, device=device, dtype=torch.bool)
        else:
            mask_tensor = torch.ones_like(slice_tensor, dtype=torch.bool)

        # Convert affine
        slice_tensor, mask_tensor, transformation = affine2transformation(
            slice_tensor, mask_tensor, resolutions, affine
        )

        slices.append(
            Slice(
                image=slice_tensor,
                mask=mask_tensor,
                transformation=transformation,
                resolution_x=resolutions[0],
                resolution_y=resolutions[1],
                resolution_z=resolutions[2],
            )
        )

    # Sort by index
    return [slice for _, slice in sorted(zip(ids, slices))]


def load_stack(
    path_vol: PathType,
    path_mask: Optional[PathType] = None,
    device: DeviceType = torch.device("cpu"),
) -> Stack:
    """
    Load Stack from NIfTI file.

    Args:
        path_vol: Path to volume NIfTI
        path_mask: Optional path to mask NIfTI
        device: Device to load onto

    Returns:
        Stack object
    """
    slices, resolutions, affine = load_nii_volume(path_vol)

    # Load or create mask
    if path_mask is None:
        mask = np.ones_like(slices, dtype=bool)
    else:
        mask, resolutions_m, affine_m = load_nii_volume(path_mask)
        mask = mask > 0

        # Verify compatibility
        if not compare_resolution_affine(
            resolutions, affine, resolutions_m, affine_m, slices.shape, mask.shape
        ):
            raise Exception(
                "Error: the sizes/resolutions/affine transformations of the "
                "input stack and stack mask do not match!"
            )

    # Convert to tensors
    slices_tensor = torch.tensor(slices, device=device)
    mask_tensor = torch.tensor(mask, device=device)

    # Convert affine
    slices_tensor, mask_tensor, transformation = affine2transformation(
        slices_tensor, mask_tensor, resolutions, affine
    )

    return Stack(
        slices=slices_tensor.unsqueeze(1),  # Add channel dim
        mask=mask_tensor.unsqueeze(1),
        transformation=transformation,
        resolution_x=resolutions[0],
        resolution_y=resolutions[1],
        thickness=resolutions[2],
        gap=resolutions[2],
        name=str(path_vol),
    )


def load_volume(
    path_vol: PathType,
    path_mask: Optional[PathType] = None,
    device: DeviceType = torch.device("cpu"),
) -> Volume:
    """
    Load Volume from NIfTI file.

    Args:
        path_vol: Path to volume NIfTI
        path_mask: Optional path to mask NIfTI
        device: Device to load onto

    Returns:
        Volume object
    """
    vol, resolutions, affine = load_nii_volume(path_vol)

    # Load or create mask
    if path_mask is None:
        mask = np.ones_like(vol, dtype=bool)
    else:
        mask, resolutions_m, affine_m = load_nii_volume(path_mask)
        mask = mask > 0

        # Verify compatibility
        if not compare_resolution_affine(
            resolutions, affine, resolutions_m, affine_m, vol.shape, mask.shape
        ):
            raise Exception(
                "Error: the sizes/resolutions/affine transformations of the "
                "input stack and stack mask do not match!"
            )

    # Convert to tensors
    vol_tensor = torch.tensor(vol, device=device)
    mask_tensor = torch.tensor(mask, device=device)

    # Convert affine
    vol_tensor, mask_tensor, transformation = affine2transformation(
        vol_tensor, mask_tensor, resolutions, affine
    )

    # Average transformation (volumes should have single transform)
    transformation = RigidTransform(transformation.axisangle().mean(0, keepdim=True))

    return Volume(
        image=vol_tensor,
        mask=mask_tensor,
        transformation=transformation,
        resolution_x=resolutions[0],
        resolution_y=resolutions[1],
        resolution_z=resolutions[2],
    )


def load_mask(path_mask: PathType, device: DeviceType = torch.device("cpu")) -> Volume:
    """
    Load mask volume.

    Convenience function where both image and mask are same data.

    Args:
        path_mask: Path to mask NIfTI
        device: Device to load onto

    Returns:
        Volume with mask data
    """
    return load_volume(path_mask, path_mask, device)
