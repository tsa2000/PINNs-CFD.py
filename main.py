"""main.py
Entry point for the PINN-CFD scaffold. Example usage:
    python main.py --train
The script will:
 - load config
 - prepare (or create demo) mesh
 - init PINN solver and train
 - save checkpoints and a small summary
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from utils import read_config, setup_logging
from mesh_handler import load_mesh, preprocess_mesh, make_demo_cube
from pinn_solver import PINNSolver
from physics_model import prepare_property_arrays, query_properties

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--demo-mesh', action='store_true', help='Create/use a tiny demo mesh')
    args = parser.parse_args()

    cfg = read_config(args.config)
    setup_logging(cfg.get('logging_level','INFO'))

    mesh_path = cfg.get('mesh_file','mesh.vtk')
    try:
        mesh = load_mesh(mesh_path)
    except Exception:
        print('Could not load mesh; creating demo cube mesh.')
        mesh = make_demo_cube(mesh_path)

    pts, cells = preprocess_mesh(mesh, normalize=True)
    coords = pts.astype(np.float32)

    # simple property arrays (uniform T,P)
    T = np.full((coords.shape[0],), float(cfg.get('temperature',300.0)))
    P = np.full((coords.shape[0],), float(cfg.get('pressure',101325.0)))
    props = prepare_property_arrays(T, P, fluid=cfg.get('fluid','Air'))

    # solver config
    solver_cfg = {
        'hidden_sizes': cfg.get('hidden_sizes',[128,128,128]),
        'learning_rate': float(cfg.get('learning_rate',1e-3)),
        'rho': float(props['rho'].mean()),
        'mu': float(props['mu'].mean()),
        'cp': float(props['cp'].mean()),
        'k_th': float(props['k_th'].mean()),
        'boundary_conditions': cfg.get('boundary_conditions',{}),
        'results_dir': cfg.get('results_dir','results'),
        'w_phys': cfg.get('w_phys',1.0),
        'w_bc': cfg.get('w_bc',10.0),
        'w_energy': cfg.get('w_energy',1.0),
    }

    solver = PINNSolver(solver_cfg, coords, cells, device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))

    if args.train:
        hist = solver.train(epochs=int(cfg.get('epochs',1000)), batch_size=int(cfg.get('batch_size',4096)))
        print('Training finished. Latest loss:', hist['loss_total'][-1])
        # save a small numpy summary
        Path(solver_cfg['results_dir']).mkdir(parents=True, exist_ok=True)
        np.save(Path(solver_cfg['results_dir']) / 'loss_history.npy', np.asarray(hist['loss_total']))
    else:
        print('Run with --train to train the PINN.')

if __name__ == '__main__':
    main()
