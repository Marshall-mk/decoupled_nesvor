"""
Hash Grid Encoding for Implicit Neural Representations

This module implements multi-resolution hash grid encoding, a key component of
Instant Neural Graphics Primitives (Instant-NGP). It provides efficient, learnable
positional encoding for coordinate-based neural networks.

Adapted from: https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/hash_encoding.py

Key Concepts:
-----------
1. **Multi-Resolution Hash Encoding**:
   - Multiple resolution levels (coarse to fine)
   - Each level has its own hash table of learnable features
   - Coordinates are hashed to table indices (handles large grids efficiently)
   - Trilinear interpolation for smooth feature lookup

2. **Why Hash Encoding?**:
   - Standard grids: memory grows as O(resolution^3) - prohibitive for high res
   - Hash grids: fixed memory O(hash_table_size), independent of resolution
   - Hash collisions are resolved naturally by gradient descent
   - Enables very high resolution encoding (up to 2048^3) in small memory

3. **How It Works**:
   - For each resolution level:
     a. Compute voxel containing the point
     b. Hash the 8 corner vertices to get feature indices
     c. Look up features from hash table
     d. Trilinearly interpolate based on point position within voxel
   - Concatenate features from all levels

Mathematical Formulation:
-----------------------
For a point x and resolution level l:
    1. Grid resolution: N_l = floor(N_0 * b^l)
       where N_0 = base_resolution, b = per_level_scale

    2. Voxel coordinates: v = floor(x * N_l)

    3. Hash function: h(v) = (⊕_i p_i * v_i) mod T
       where ⊕ is XOR, p_i are primes, T = 2^log2_hashmap_size

    4. Feature lookup: f_l(x) = trilinear_interp(hash_table_l[h(v + offset)])
       for offset in {0,1}^3 (8 corners)

    5. Final encoding: concat([f_0(x), f_1(x), ..., f_{L-1}(x)])

Classes:
-------
HashEmbedder: Multi-resolution hash grid encoder
    - __init__: Initialize hash tables and parameters
    - forward: Encode 3D coordinates to feature vectors
    - get_voxel_vertices: Compute voxel corners and hash to indices
    - trilinear_interp: Interpolate features from 8 corners

Functions:
---------
_hash: Spatial hash function using prime number XOR trick
"""

from typing import Tuple
import torch
import torch.nn as nn


class HashEmbedder(nn.Module):
    """
    Multi-resolution hash grid encoder for 3D coordinates.

    This encoder maps 3D coordinates to a high-dimensional feature vector by:
    1. Querying multiple resolution levels (coarse to fine)
    2. Using spatial hashing to map coordinates to learnable features
    3. Trilinear interpolation for smooth feature lookup
    4. Concatenating features from all levels

    The hash grid enables very high resolution encoding with bounded memory:
    - Memory: O(L * T * F) where L=levels, T=hash_table_size, F=features_per_level
    - Independent of actual grid resolution (can be 2048^3 or higher!)

    Attributes:
        n_levels: Number of resolution levels (typically 16)
        n_features_per_level: Features per level (typically 2)
        log2_hashmap_size: log2 of hash table size (typically 19, i.e., 2^19 = 512K entries)
        base_resolution: Coarsest resolution (typically 16)
        b: Per-level scale factor (typically 1.39)
        embeddings: ModuleList of embedding tables (one per level)
        box_offsets: (1, 8, 3) tensor with 8 corners of unit cube

    Example:
        >>> encoder = HashEmbedder(
        ...     n_input_dims=3,
        ...     n_levels=16,
        ...     n_features_per_level=2,
        ...     log2_hashmap_size=19,
        ...     base_resolution=16,
        ...     per_level_scale=1.39
        ... )
        >>> xyz = torch.randn(4096, 3)  # Batch of 3D coordinates
        >>> features = encoder(xyz)  # (4096, 32) = 16 levels * 2 features/level
        >>> print(features.shape)  # torch.Size([4096, 32])
    """

    def __init__(
        self,
        n_input_dims: int = 3,
        otype: str = "HashGrid",
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.39,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize hash grid encoder.

        Args:
            n_input_dims: Input dimension (must be 3 for 3D coordinates)
            otype: Encoding type (must be "HashGrid")
            n_levels: Number of resolution levels (more = finer details)
            n_features_per_level: Features per level (more = more capacity per level)
            log2_hashmap_size: log2 of hash table size (19 = 512K entries)
            base_resolution: Coarsest grid resolution
            per_level_scale: Scale factor between levels (typically ~1.4)
            dtype: Data type for features (float32 or float16)

        Note:
            - Total output dimension = n_levels * n_features_per_level
            - Finest resolution = base_resolution * per_level_scale^(n_levels-1)
            - Memory usage = n_levels * 2^log2_hashmap_size * n_features_per_level * bytes_per_element
        """
        super(HashEmbedder, self).__init__()
        assert n_input_dims == 3 and otype == "HashGrid"

        # Store hyperparameters
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.b = per_level_scale  # Scale factor between levels

        # Create hash tables (embedding tables) for each level
        # Each table has 2^log2_hashmap_size entries, each with n_features_per_level dimensions
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(2**self.log2_hashmap_size, self.n_features_per_level)
                for _ in range(n_levels)
            ]
        )

        # Initialize embeddings with small random values
        # Small initialization is important for stable training
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

        # Pre-compute 8 corners of unit cube: {0,1}^3
        # Used to get voxel corner vertices for trilinear interpolation
        # Order: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
        self.register_buffer(
            "box_offsets",
            torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]]),
        )

    def trilinear_interp(
        self,
        x: torch.Tensor,
        voxel_min_vertex: torch.Tensor,
        voxel_embedds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Trilinear interpolation of features from 8 voxel corners.

        Trilinear interpolation is a 3D extension of bilinear interpolation.
        It smoothly interpolates values within a voxel based on the point's
        position relative to the voxel corners.

        Algorithm (from Wikipedia):
        1. Interpolate along x-axis (4 pairs) -> 4 values
        2. Interpolate along y-axis (2 pairs) -> 2 values
        3. Interpolate along z-axis (1 pair) -> 1 value

        Args:
            x: (B, 3) 3D coordinates within the voxel
            voxel_min_vertex: (B, 3) coordinates of minimum corner (0,0,0) vertex
            voxel_embedds: (B, 8, F) features at 8 corners
                Corner order: (0,0,0), (0,0,1), (0,1,0), (0,1,1),
                             (1,0,0), (1,0,1), (1,1,0), (1,1,1)

        Returns:
            (B, F) interpolated features

        Note:
            - Weights are fractional coordinates within voxel: w = x - voxel_min_vertex
            - Each weight in [0, 1] representing position within voxel
            - Result is smooth and differentiable (important for gradient-based learning)
        """
        # Compute interpolation weights (fractional position within voxel)
        # weights[i] = 0 means at minimum corner, 1 means at maximum corner
        weights = x - voxel_min_vertex  # (B, 3)

        # Step 1: Interpolate along x-axis (dimension 0)
        # Interpolate between (0,0,0)↔(1,0,0), (0,0,1)↔(1,0,1), (0,1,0)↔(1,1,0), (0,1,1)↔(1,1,1)
        c00 = (
            voxel_embedds[:, 0] * (1 - weights[:, 0][:, None])  # Weight for x=0
            + voxel_embedds[:, 4] * weights[:, 0][:, None]  # Weight for x=1
        )
        c01 = (
            voxel_embedds[:, 1] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 5] * weights[:, 0][:, None]
        )
        c10 = (
            voxel_embedds[:, 2] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 6] * weights[:, 0][:, None]
        )
        c11 = (
            voxel_embedds[:, 3] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 7] * weights[:, 0][:, None]
        )

        # Step 2: Interpolate along y-axis (dimension 1)
        # Interpolate between c00↔c10 and c01↔c11
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # Step 3: Interpolate along z-axis (dimension 2)
        # Final interpolation between c0↔c1
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c  # (B, F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D coordinates using multi-resolution hash grid.

        For each resolution level:
        1. Scale coordinates to that level's resolution
        2. Find voxel containing the point
        3. Hash 8 corner vertices to get feature indices
        4. Look up features from hash table
        5. Trilinearly interpolate based on point position

        Then concatenate features from all levels.

        Args:
            x: (B, 3) 3D coordinates to encode
               Assumed to be in [0, 1]^3 range (normalized by bounding box)

        Returns:
            (B, n_levels * n_features_per_level) encoded features

        Example:
            >>> encoder = HashEmbedder(n_levels=16, n_features_per_level=2)
            >>> xyz = torch.rand(1000, 3)  # Random points in [0, 1]^3
            >>> features = encoder(xyz)
            >>> print(features.shape)  # (1000, 32)
        """
        # List to accumulate features from all levels
        x_embedded_all = []

        # Process each resolution level
        for i in range(self.n_levels):
            # Compute resolution for this level: N_l = N_0 * b^l
            resolution = int(self.base_resolution * self.b**i)

            # Get voxel information for this level
            (
                voxel_min_vertex,  # (B, 3) minimum corner of voxel
                hashed_voxel_indices,  # (B, 8) hashed indices for 8 corners
                xi,  # (B, 3) scaled coordinates
            ) = self.get_voxel_vertices(x, resolution)

            # Look up features from hash table for 8 corners
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)  # (B, 8, F)

            # Trilinearly interpolate features
            x_embedded = self.trilinear_interp(
                xi, voxel_min_vertex, voxel_embedds
            )  # (B, F)

            # Append to list
            x_embedded_all.append(x_embedded)

        # Concatenate features from all levels
        return torch.cat(x_embedded_all, dim=-1)  # (B, L*F)

    def get_voxel_vertices(
        self, xyz: torch.Tensor, resolution: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get voxel corners and hash them to feature indices.

        For a given point and resolution:
        1. Scale coordinates to grid space: xyz_grid = xyz * resolution
        2. Find voxel: voxel_min = floor(xyz_grid)
        3. Compute 8 corner coordinates: voxel_min + {0,1}^3
        4. Hash corners to indices in hash table

        Args:
            xyz: (B, 3) coordinates in [0, 1]^3
            resolution: Grid resolution for this level

        Returns:
            voxel_min_vertex: (B, 3) minimum corner coordinates (as integers)
            hashed_voxel_indices: (B, 8) hash table indices for 8 corners
            xyz: (B, 3) scaled coordinates (xyz * resolution)

        Note:
            - voxel_min_vertex is the (0,0,0) corner of the voxel
            - The 8 corners are voxel_min + box_offsets
            - Hashing maps potentially huge grid indices to fixed-size hash table
        """
        # Scale coordinates to grid resolution
        xyz = xyz * resolution  # (B, 3)

        # Find voxel containing point (floor to get minimum corner)
        voxel_min_vertex = torch.floor(xyz).int()  # (B, 3)

        # Compute 8 corner vertices by adding box_offsets {0,1}^3
        # Broadcasting: (B, 3) + (1, 8, 3) -> (B, 8, 3)
        voxel_indices = voxel_min_vertex.unsqueeze(1) + self.box_offsets

        # Hash corner coordinates to indices in hash table
        hashed_voxel_indices = _hash(voxel_indices, self.log2_hashmap_size)  # (B, 8)

        return voxel_min_vertex, hashed_voxel_indices, xyz


def _hash(coords: torch.Tensor, log2_hashmap_size: int) -> torch.Tensor:
    """
    Spatial hash function for mapping 3D grid coordinates to hash table indices.

    Uses the XOR-based hash function with prime numbers, a standard technique
    in spatial hashing. This distributes coordinates uniformly across the hash
    table, minimizing collisions.

    Hash function: h(c) = (⊕_i p_i * c_i) mod T
    where:
        - c_i are coordinate components
        - p_i are large prime numbers
        - ⊕ is XOR operation
        - T = 2^log2_hashmap_size (hash table size)

    Args:
        coords: (..., D) integer coordinates to hash
                Can be up to 7 dimensions, typically 3D
        log2_hashmap_size: log2 of hash table size

    Returns:
        (...,) hashed indices in range [0, 2^log2_hashmap_size - 1]

    Note:
        - Primes are chosen to be large (~2^31) for good distribution
        - XOR provides fast, non-commutative mixing of coordinates
        - Modulo by power of 2 (using bitwise AND) is very fast
        - Hash collisions are handled by learning (gradient descent averages)

    Example:
        >>> coords = torch.tensor([[0, 0, 0], [1, 2, 3], [100, 200, 300]])
        >>> indices = _hash(coords, log2_hashmap_size=10)  # Hash to 1024 entries
        >>> print(indices.shape)  # (3,)
        >>> print(indices.max() < 1024)  # True
    """
    # Large prime numbers for each dimension (up to 7D)
    # These primes are chosen to minimize hash collisions
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    # Initialize XOR result with zeros
    xor_result = torch.zeros_like(coords)[..., 0]

    # XOR combination of (prime * coordinate) for each dimension
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    # Modulo hash table size using bitwise AND (fast for powers of 2)
    # (1 << log2_hashmap_size) - 1 creates a bit mask: 2^N - 1
    # E.g., for N=10: 1111111111 in binary = 1023 in decimal
    return (
        torch.tensor((1 << log2_hashmap_size) - 1, device=xor_result.device)
        & xor_result
    )
