"""pinn_solver.py
High-level PINN solver class using PyTorch.
The solver integrates property surrogate support, adaptive sampling stub, BC enforcement
and export helpers. The core residual functions are implemented in a differentiable way.
"""
import os
from pathlib import Path
import logging
import numpy as np
try:
    import torch
    import torch.nn as nn
    from torch.autograd import grad
except Exception:
    raise RuntimeError('PyTorch is required for pinn_solver.py')

from physics_model import query_properties

class MLP(nn.Module):
    def __init__(self, in_dim=3, hidden=(128,128,128), out_dim=5):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PINNSolver:
    def __init__(self, cfg, nodes, cells, device=None, surrogate=None, tb_writer=None):
        self.cfg = cfg
        self.nodes = nodes.astype(np.float32)
        self.cells = cells
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.surrogate = surrogate
        self.tb = tb_writer
        self.model = MLP(in_dim=3, hidden=tuple(cfg.get('hidden_sizes',[128,128,128])), out_dim=5).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.get('learning_rate',1e-3))
        self.loss_fn = nn.MSELoss()
        logging.info(f'PINNSolver initialized on device {self.device}')

    def _random_inside(self, n):
        mins = self.nodes.min(axis=0)
        maxs = self.nodes.max(axis=0)
        pts = np.random.rand(n,3) * (maxs - mins) + mins
        return torch.tensor(pts, dtype=torch.float32, device=self.device, requires_grad=True)

    def compute_residuals(self, x):
        # x: (N,3)
        x = x.clone().requires_grad_(True)
        out = self.model(x)
        u = out[:,0:1]; v = out[:,1:2]; w = out[:,2:3]; p = out[:,3:4]; T = out[:,4:5]

        def grads(f):
            g = grad(f.sum(), x, create_graph=True)[0]
            return g[:,0:1], g[:,1:2], g[:,2:3]

        u_x, u_y, u_z = grads(u)
        v_x, v_y, v_z = grads(v)
        w_x, w_y, w_z = grads(w)
        p_x, p_y, p_z = grads(p)
        T_x, T_y, T_z = grads(T)

        # properties (use surrogate if provided)
        if self.surrogate is not None:
            TP = torch.cat([T, p], dim=1)
            props = self.surrogate(TP)
            rho = props[:,0:1]; mu = props[:,1:2]; cp = props[:,2:3]; k_th = props[:,3:4]
        else:
            # simple constant fallback (differentiable)
            rho = torch.ones_like(u) * float(self.cfg.get('rho',1.0))
            mu = torch.ones_like(u) * float(self.cfg.get('mu',1e-5))
            cp = torch.ones_like(u) * float(self.cfg.get('cp',1000.0))
            k_th = torch.ones_like(u) * float(self.cfg.get('k_th',0.026))

        # continuity (incompressible-like)
        cont = u_x + v_y + w_z

        # convective terms
        conv_x = rho * (u * u_x + v * u_y + w * u_z)
        conv_y = rho * (u * v_x + v * v_y + w * v_z)
        conv_z = rho * (u * w_x + v * w_y + w * w_z)

        # Laplacian-like viscous (approx)
        def laplacian(comp):
            gx, gy, gz = grads(comp)
            gxx = grad(gx.sum(), x, create_graph=True)[0][:,0:1]
            gyy = grad(gy.sum(), x, create_graph=True)[0][:,1:2]
            gzz = grad(gz.sum(), x, create_graph=True)[0][:,2:3]
            return gxx + gyy + gzz

        lap_u = laplacian(u); lap_v = laplacian(v); lap_w = laplacian(w)

        mom_x = conv_x + p_x - mu * lap_u
        mom_y = conv_y + p_y - mu * lap_v
        mom_z = conv_z + p_z - mu * lap_w

        adv_T = rho * cp * (u * T_x + v * T_y + w * T_z)
        kTx = k_th * T_x; kTy = k_th * T_y; kTz = k_th * T_z
        div_kT = grad(kTx.sum(), x, create_graph=True)[0][:,0:1] + grad(kTy.sum(), x, create_graph=True)[0][:,1:2] + grad(kTz.sum(), x, create_graph=True)[0][:,2:3]
        energy = adv_T - div_kT

        return cont, [mom_x, mom_y, mom_z], energy

    def bc_loss(self, n=512):
        # Simple Dirichlet BC sampling from config (face based). For demo we support 'inlet' with velocity and 'outlet' pressure.
        total = torch.tensor(0.0, device=self.device)
        bcs = self.cfg.get('boundary_conditions',{})
        if not bcs:
            return total
        for name, bc in bcs.items():
            face = bc.get('face')
            field = bc.get('field')
            val = bc.get('value')
            # sample points on face: simple heuristic using nodes
            pts = self._random_inside(min(n, 256))
            out = self.model(pts)
            if field == 'velocity':
                tgt = torch.tensor(val, dtype=torch.float32, device=self.device).reshape(1,3).expand(pts.shape[0],3)
                pred = out[:,0:3]
                total = total + self.loss_fn(pred, tgt)
            elif field == 'pressure':
                pred = out[:,3:4]
                tgt = torch.full_like(pred, float(val), device=self.device)
                total = total + self.loss_fn(pred, tgt)
            elif field == 'temperature':
                pred = out[:,4:5]
                tgt = torch.full_like(pred, float(val), device=self.device)
                total = total + self.loss_fn(pred, tgt)
        return total

    def train(self, epochs=1000, batch_size=4096, checkpoint_dir=None):
        history = {'loss_total':[]}
        checkpoint_dir = Path(checkpoint_dir or self.cfg.get('results_dir','results'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for ep in range(1, epochs+1):
            coll = self._random_inside(batch_size)
            cont, moms, energy = self.compute_residuals(coll)
            loss_phys = self.loss_fn(cont, torch.zeros_like(cont))
            for m in moms:
                loss_phys = loss_phys + self.loss_fn(m, torch.zeros_like(m))
            loss_energy = self.loss_fn(energy, torch.zeros_like(energy))
            loss_bc = self.bc_loss(n=min(512, batch_size//4))
            w_phys = float(self.cfg.get('w_phys',1.0))
            w_bc = float(self.cfg.get('w_bc',10.0))
            w_energy = float(self.cfg.get('w_energy',1.0))
            loss = w_phys*loss_phys + w_bc*loss_bc + w_energy*loss_energy
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            history['loss_total'].append(float(loss.item()))
            if ep % max(1, epochs//10) == 0 or ep==1:
                logging.info(f"[train] ep {ep}/{epochs} loss={loss.item():.4e}")
                torch.save({'epoch':ep,'state':self.model.state_dict()}, checkpoint_dir / f'pinn_ep{ep}.pt')
        return history

    def predict_on_points(self, points):
        self.model.eval()
        pts = torch.tensor(points, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(pts).cpu().numpy()
        return out
