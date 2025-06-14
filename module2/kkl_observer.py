# -*- coding: utf-8 -*-
"""Physics-informed KKL observer for a 4-state slice of World3."""
import torch, torch.nn as nn, numpy as np

__all__ = ["KKLObserver"]

# --------------------------------------------------------------------- #
# Small helper nets
# --------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                    # x shape (N , in_dim)
        return self.net(x)

# --------------------------------------------------------------------- #
# Main observer
# --------------------------------------------------------------------- #
class KKLObserver:
    r"""
    Learned KKL observer   (slides 30-31)

        x_dot = f(x),  y = h(x)

        z_dot = A*z + B*y               (observer copy, A Hurwitz)
        x_hat = T_eta*(z)               (inverse net)
    """

    # ------------- Initialisation ------------------------------------------
    def __init__(self, *,
                 A: np.ndarray,
                 B: np.ndarray,
                 C: np.ndarray,
                 state_names,
                 meas_names,
                 L_gain: float | None = None,
                 v_phys: float = 1e-2,
                 device: str = "cpu"):
        self.device = device
        # Fixed linear part
        self.A = torch.tensor(A, dtype=torch.float32, device=device)
        self.B = torch.tensor(B, dtype=torch.float32, device=device)
        self.C = torch.tensor(C, dtype=torch.float32, device=device)
        z_dim, y_dim = self.A.shape[0], self.C.shape[0]

        # Learnable forward and inverse maps
        self.T = MLP(len(state_names), z_dim).to(device)        # theta
        self.T_inv = MLP(z_dim, len(state_names)).to(device)    # eta

        self.state_names = state_names
        self.meas_names = meas_names
        self.v_phys = v_phys

        self.z = None            # Observer internal state (torch tensor)

        # To decide whether to use contracting copy or Luenberger observer
        self.use_luenberger = L_gain is not None
        if self.use_luenberger:
            self.L = (L_gain *
                      torch.eye(z_dim, y_dim, dtype=torch.float32, device=device))
        else:
            self.L = None            # keeps old behaviour

    # ------------- Stage 1: learn T(x) -------------------------------------
    def fit_T(self, X, Y, *, epochs=400, lr=2e-3, t_trunc=20):
        """
        Learn theta: minimise  data-loss + v * physics-loss (slides 30 & 28).
        X : ndarray (N , n_x)  - states
        Y : ndarray (N , n_y)  - measured outputs (h(x))
        """
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device, requires_grad=True)
        Y_torch = torch.tensor(Y, dtype=torch.float32, device=self.device)

        # Pre-compute x_dot via finite diff (shape (N-1 , n_x))
        dX = torch.tensor(np.diff(X, axis=0), dtype=torch.float32, device=self.device)
        X_mid = X_torch[:-1]   # align shapes
        Y_mid = Y_torch[:-1]

        opt = torch.optim.Adam(self.T.parameters(), lr=lr)
        N   = X_mid.shape[0]

        for k in range(epochs):
            opt.zero_grad()
            Z_pred = self.T(X_mid)                      # (N , z_dim)

            # ---------- Data loss (after truncation) ----------
            data_loss = torch.mean((Z_pred[t_trunc:] - self._z_traj(X_mid, Y_mid)[t_trunc:])**2)

            # ---------- Physics loss ----------
            J_cols = []
            for j in range(Z_pred.shape[1]):            # loop over z components
                grads = torch.autograd.grad(Z_pred[:, j].sum(), X_mid, create_graph=True)[0]
                J_cols.append((grads * dX).sum(dim=1, keepdim=True))
            Tdot = torch.cat(J_cols, dim=1)             # (N-1 , z_dim)
            phys_loss = torch.mean((Tdot - (Z_pred @ self.A.T + Y_mid @ self.B.T))**2)

            loss = data_loss + self.v_phys * phys_loss
            loss.backward();  opt.step()

            if k % 50 == 0:
                print(f"[T-train] {k:3d}:  data {data_loss.item():.3e}   phys {phys_loss.item():.3e}")

    # ------------- Stage 2: learn inverse ----------------------------------
    def fit_Tinv(self, X, *, epochs=300, lr=2e-3):
        """Minimise ‖x - T_inv(T(x))‖^2 (slide 31)."""
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self.T_inv.parameters(), lr=lr)

        for k in range(epochs):
            opt.zero_grad()
            Z = self.T(X_torch).detach()                # freeze T
            X_rec = self.T_inv(Z)
            inv_loss = torch.mean((X_rec - X_torch)**2)
            inv_loss.backward(); opt.step()
            if k % 50 == 0:
                print(f"[T*-train] {k:3d}:  inv-loss {inv_loss.item():.3e}")

    # ------------- On-line observer step -----------------------------------
    @torch.no_grad()
    def step(self, y, *args):
        """
        Forward Euler integration of z_dot = A*z + B*y ; returns numpy x_hat.
        
        Call patterns
        ------------
        (y, dt)            -> contracting-copy  ( B multiplies y )
        (y, u, dt)         -> Luenberger        ( B multiplies u,  L multiplies y-Cz )
        """
        # ---------------- Argument parsing ----------------
        if len(args) == 1:          # legacy (y, dt)
            u  = None
            dt = args[0]
        elif len(args) == 2:        # extended (y, u, dt)
            u, dt = args
        else:
            raise ValueError("step expects (y, dt) or (y, u, dt)")
        
        # ---------------- To tensors ----------------
        y  = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        if u is None:
            u = torch.zeros(self.B.shape[1], dtype=torch.float32, device=self.device)
        else:
            u = torch.as_tensor(u, dtype=torch.float32, device=self.device)

        # -------------- First-call initialisation -------------
        if self.z is None:          # initialise
            self.z = torch.zeros(self.A.shape[0], device=self.device)

        # -------------- Euler update --------------------------
        if self.use_luenberger:
            ez   = y - (self.C @ self.z)
            self.z = self.z + dt * (self.A @ self.z + self.B @ u + self.L @ ez)
        else:                       # old contracting copy
            self.z = self.z + dt * (self.A @ self.z + self.B @ y)

        # -------------- Decode -------------------------------
        x_hat = self.T_inv(self.z.unsqueeze(0)).squeeze(0)  # (n_x,)
        return x_hat.detach().cpu().numpy()

    # ------------- Helper: simulate observer copy --------------------------
    @torch.no_grad()
    def _z_traj(self, X_mid, Y_mid):
        """
        Return z[k] obtained from Euler-integrating z_dot = A z + B y
        along the measured-output trajectory Y_mid (same length as X_mid).
        """
        z  = torch.zeros(self.A.shape[0], device=self.device)   # (n_z,)
        Zs = []                                                 # list to store each step

        for k in range(X_mid.shape[0]):
            z = z + (self.A @ z + self.B @ Y_mid[k])            # dt = 1 here
            Zs.append(z.clone())

        return torch.stack(Zs, dim=0)                           # shape (N , n_z)