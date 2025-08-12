"""mesh_handler.py
Utilities to load, save and preprocess meshes using meshio.
This module intentionally keeps things simple so you can extend it to your needs.
"""
import numpy as np
try:
    import meshio
except Exception:
    meshio = None

def ensure_meshio():
    if meshio is None:
        raise RuntimeError("meshio is required for mesh handling. Install with `pip install meshio`")

def load_mesh(path):
    """Load a mesh file (supports many formats via meshio)"""
    ensure_meshio()
    return meshio.read(path)

def save_mesh(mesh, path):
    """Write mesh to path using meshio."""
    ensure_meshio()
    meshio.write(path, mesh)

def preprocess_mesh(mesh, normalize=True):
    """Convert mesh to numpy arrays (points, cells) and optionally normalize coordinates.
    Returns (points, cells) where points is (N,3) float64 and cells is a list of (cell_type, connectivity).
    """
    ensure_meshio()
    pts = np.asarray(mesh.points, dtype=np.float64)
    cells = []
    for block in mesh.cells:
        # meshio.CellBlock or tuple
        if hasattr(block, 'type') and hasattr(block, 'data'):
            cells.append((block.type, np.asarray(block.data, dtype=np.int32)))
        else:
            t, d = block
            cells.append((t, np.asarray(d, dtype=np.int32)))
    if normalize:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        span = (maxs - mins)
        span[span == 0] = 1.0
        pts = (pts - mins) / span
    return pts, cells

def make_demo_cube(path=None):
    """Create and return a tiny demo tetrahedral cube mesh (requires meshio write)"""
    ensure_meshio()
    pts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
    # a few tets (toy)
    cells = [meshio.CellBlock('tetra', np.array([[0,1,2,5],[0,2,3,7],[0,5,6,7],[0,4,5,7],[0,2,5,7]]))]
    m = meshio.Mesh(points=pts, cells=cells)
    if path is not None:
        meshio.write(path, m)
    return m
