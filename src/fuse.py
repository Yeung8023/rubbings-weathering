"""Multi-epoch, multi-character rubbing fusion.

The statistical design is what makes the problem well posed.  For one stele we
observe n rubbings; each rubbing shows C characters.  Then

  * the impression style theta_i = (lam, eps, s, alpha) is **shared by all C
    characters of rubbing i** and independent across i  -- 4n unknowns,
  * the weathering state (sigma_j, kappa_j) is **shared by all C characters at
    epoch j** and monotone in j                          -- 2n unknowns,
  * the pristine relief h0_c is per character            -- C fields,
  * spalling and paper deformation are per (character, rubbing).

So nC images constrain 6n global nuisance parameters.  With C ~ 10-100 the
nuisance is pinned down by the data, and the monotone common component is
separated from the independent per-rubbing component -- see identify.py for
the quantitative version of this argument.

Gauge fixing
------------
Three exact gauge freedoms are removed by construction:
  1. time origin      sigma_0 = 0, kappa_0 = 1, spall_0 = 0  (the earliest
     rubbing defines the reference state of the stone);
  2. spatial frame    the warp of rubbing 0 is fixed to identity, so h0 lives
     in the frame of the earliest rubbing;
  3. depth scale      the 99.5th percentile of h0 is anchored to the measured
     carving depth of the stele (a quantity obtainable from the extant stone).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import torchmodel as T


def _sp(x):
    return F.softplus(x)


class Fusion(torch.nn.Module):
    """images: (C, n, H, W)."""

    def __init__(self, C, n, H, Wd, px_mm, device="cuda", spall=True,
                 warp_ctrl=8, dtype=torch.float32, dt=None,
                 relief="levelset", depth_mm=1.4, half_width_mm=0.55,
                 tau=0.6, spall_stride=8, spall_model="levelset",
                 spall_soft=0.35, fix_a=None, fix_b=None, gain=False):
        """dt: (n,) elapsed centuries since the stele was carved.  If given,
        the weathering trajectory is the two-parameter rate law
            sigma(t) = a * t,   kappa(t) = exp(-b * t),
        so the recovered relief h0 is the *original carving* rather than the
        state at the earliest surviving impression.  If None, the trajectory
        is free (monotone increments) and h0 is the state at rubbing 0."""
        super().__init__()
        self.C, self.n, self.H, self.W, self.px_mm = C, n, H, Wd, px_mm
        self.use_spall = spall
        self.register_buffer("dt", None if dt is None
                             else torch.as_tensor(dt, dtype=dtype))
        self.relief = relief
        self.depth_mm, self.half_width_mm, self.tau = depth_mm, half_width_mm, tau
        self.spall_soft = spall_soft
        self.fix_a, self.fix_b = fix_a, fix_b
        P = lambda t: torch.nn.Parameter(t)
        self.a_h0 = P(torch.zeros(C, 1, H, Wd, dtype=dtype))
        # shared impression style, one per rubbing
        self.a_lam = P(torch.full((n,), 0.0, dtype=dtype))
        self.a_eps = P(torch.full((n,), -2.0, dtype=dtype))
        self.a_s = P(torch.full((n,), -3.0, dtype=dtype))
        self.a_alpha = P(torch.full((n,), 1.5, dtype=dtype))
        # shared monotone weathering, one per epoch
        self.d_sigma = P(torch.full((n,), -4.0, dtype=dtype))
        self.d_kappa = P(torch.full((n,), -4.0, dtype=dtype))
        # per (character, epoch).  Stone flakes are millimetre-to-centimetre
        # features, so the spall increments live on a coarse grid and are
        # upsampled: at full pixel resolution this field would simply absorb
        # paper texture and staining and corrupt everything else.
        self.spall_stride = spall_stride
        self.spall_model = spall_model
        hs, ws = max(4, H // spall_stride), max(4, Wd // spall_stride)
        if spall_model == "levelset":
            # one time-invariant potential per character; the flakes at epoch j
            # are its super-level set at a shared, monotonically falling
            # threshold.  Nested by construction, and it mirrors the way stone
            # actually loses flakes: the weak patches were always the weak
            # patches, they simply reach the surface at different times.
            self.spall_pot = P(0.05 * torch.randn(C, 1, hs, ws, dtype=dtype))
            th = torch.full((n,), -2.0, dtype=dtype)
            th[0] = 3.0                       # start with essentially no flaking
            self.d_thresh = P(th)
            self.a_spall_depth = P(torch.tensor(0.5, dtype=dtype))
            self.d_spall = P(torch.zeros(1, dtype=dtype))      # unused
        else:
            self.d_spall = P(torch.full((C, n, hs, ws), -6.0, dtype=dtype))
        self.wctrl = P(torch.zeros(C, n, 2, warp_ctrl, warp_ctrl, dtype=dtype))
        # rate-law parameters (used when dt is given)
        self.a_rate = P(torch.tensor(-2.5, dtype=dtype))   # sigma per century
        self.b_rate = P(torch.tensor(-2.5, dtype=dtype))   # kappa decay /century
        # per-image paper base tone.  For museum photographs we also allow a
        # per-impression photometric gain: different sheets were digitised on
        # different days with different exposure, and without a gain the ink
        # density alpha has to absorb it and the physics is distorted.  The
        # price is that alpha is then identified only up to that gain, which we
        # state rather than hide.
        self.b = P(torch.zeros(C, n, dtype=dtype))
        self.log_g = P(torch.zeros(n, dtype=dtype))
        self.use_gain = gain
        self.to(device)
        self.device = device

    # ---- derived ----------------------------------------------------------
    def h0(self, idx=None):
        """Pristine relief.

        ``relief='levelset'`` (default) parameterises the *carved region* by a
        level set and derives the depth field from its distance transform, so
        the unknown is a shape rather than an arbitrary image.  That is the
        same generative model the stone was cut with -- sharp boundaries, a
        V-section, thin strokes shallower than thick ones -- and it keeps
        deconvolution from drifting into smooth grey mush.
        ``relief='free'`` is the unconstrained alternative used in the ablation.
        """
        a = self.a_h0 if idx is None else self.a_h0[idx]
        if self.relief == "levelset":
            h, _ = T.relief_from_levelset(a, self.px_mm, self.depth_mm,
                                          self.half_width_mm, self.tau)
            return h
        return _sp(a)

    def fg(self, idx=None):
        a = self.a_h0 if idx is None else self.a_h0[idx]
        return torch.sigmoid(a / self.tau) if self.relief == "levelset" \
            else (_sp(a) > 0.5 * self.depth_mm).float()

    def styles(self):
        lam = 0.15 + 1.85 * torch.sigmoid(self.a_lam)      # 0.15 .. 2.00 mm
        eps = 0.03 + 0.27 * torch.sigmoid(self.a_eps)      # 0.03 .. 0.30 mm
        s = 0.015 + 0.135 * torch.sigmoid(self.a_s)        # 0.015.. 0.15 mm
        alpha = 0.35 + 0.64 * torch.sigmoid(self.a_alpha)  # 0.35 .. 0.99
        return lam, eps, s, alpha

    def rates(self):
        a = self.fix_a if self.fix_a is not None else _sp(self.a_rate)
        b = self.fix_b if self.fix_b is not None else _sp(self.b_rate)
        return a, b

    def sigma(self):
        if self.dt is not None:
            return self.rates()[0] * self.dt
        inc = _sp(self.d_sigma)
        return torch.cumsum(inc, 0) - inc[0]               # sigma_0 = 0

    def kappa(self):
        if self.dt is not None:
            return torch.exp(-self.rates()[1] * self.dt)
        inc = _sp(self.d_kappa)
        return torch.exp(-(torch.cumsum(inc, 0) - inc[0]))  # kappa_0 = 1

    def spall(self, idx=None):
        """Nested pit fields (mm of extra depth)."""
        if not self.use_spall:
            return None
        if self.spall_model == "levelset":
            pot = self.spall_pot if idx is None else self.spall_pot[idx]
            C = pot.shape[0]
            # thresholds fall monotonically, so the flake supports are nested
            thr = self.d_thresh[0] - torch.cumsum(_sp(self.d_thresh), 0) \
                + _sp(self.d_thresh)[0]
            g = torch.sigmoid((pot - thr.view(1, -1, 1, 1)) / self.spall_soft)
            c = _sp(self.a_spall_depth) * g
        else:
            inc = _sp(self.d_spall)
            c = inc.cumsum(1) if self.dt is not None else inc.cumsum(1) - inc[:, 0:1]
            if idx is not None:
                c = c[idx]
            C = c.shape[0]
        n = c.shape[1]
        c = F.interpolate(c.reshape(C * n, 1, *c.shape[-2:]),
                          size=(self.H, self.W), mode="bilinear",
                          align_corners=False)
        return c.reshape(C, n, self.H, self.W)

    def flow(self, idx=None):
        w = self.wctrl if idx is None else self.wctrl[idx]
        C = w.shape[0]
        f = F.interpolate(w.reshape(C * self.n, 2, *w.shape[-2:]),
                          size=(self.H, self.W), mode="bicubic", align_corners=True)
        f = f.reshape(C, self.n, 2, self.H, self.W)
        mask = torch.ones(self.n, 1, 1, 1, device=f.device, dtype=f.dtype)
        mask[0] = 0.0                                       # rubbing 0 = identity
        return f * mask

    # ---- forward ----------------------------------------------------------
    def forward(self, idx=None):
        """idx: optional index tensor selecting a subset of characters, so a
        large corpus can be fitted by gradient accumulation over chunks."""
        lam, eps, s, alpha = self.styles()
        sig, kap = self.sigma(), self.kappa()
        n = self.n
        h0 = self.h0(idx)
        C = h0.shape[0]
        sp = self.spall(idx)
        h = h0.repeat_interleave(n, 0)                      # (C*n,1,H,W)
        h = T.gauss_blur_t(h, sig.repeat(C) / self.px_mm)
        h = kap.repeat(C).view(-1, 1, 1, 1) * h
        if sp is not None:
            h = h + sp.reshape(C * n, 1, self.H, self.W)
        y = T.render_t(h, lam.repeat(C), eps.repeat(C), s.repeat(C),
                       alpha.repeat(C), self.px_mm)
        y = T.warp_t(y, self.flow(idx).reshape(C * n, 2, self.H, self.W))
        if self.use_gain:
            y = torch.exp(self.log_g).repeat(C).view(-1, 1, 1, 1) * y
        bb = self.b if idx is None else self.b[idx]
        y = y + bb.reshape(-1, 1, 1, 1)
        return y.reshape(C, n, self.H, self.W), h


def robust(res, c):
    """Cauchy (Lorentzian) loss.

    Each sheet carries losses of its own -- torn paper backed with blank silk,
    spilled or retouched ink -- which no shared latent relief can explain.  A
    quadratic data term would let those outliers drag the reconstruction; the
    Cauchy loss caps their influence at a residual of about ``c``.
    """
    return (0.5 * c ** 2 * torch.log1p((res / c) ** 2)).mean()


def tv(x):
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + \
           (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def fit(images, px_mm, depth_anchor_mm=1.4, sizes=(128, 256), iters=(700, 900),
        lr=0.06, device="cuda", spall=True, w_tv=1.5e-3, w_spall=2e-3,
        w_anchor=2.0, w_flow=1e-3, w_tone=3.0, warp=True, verbose=True, seed=0,
        dt=None, chunk=6, relief="free", half_width_mm=0.55, tau=0.6,
        w_perim=4e-3, w_dw=0.0, spall_stride=8, huber_c=0.15,
        spall_model="levelset", w_style=0.0, fix_a=None, fix_b=None,
        fix_styles=None, return_loss=False, gain=False, w_gain=1.0):
    """images: (C, n, H, W) in [0,1].  Returns recovered relief + trajectory."""
    torch.manual_seed(seed)
    C, n = images.shape[:2]
    Y_full = torch.tensor(images, dtype=torch.float32, device=device)
    model = None
    for sz, it in zip(sizes, iters):
        Y = F.interpolate(Y_full.reshape(C * n, 1, *images.shape[-2:]),
                          size=(sz, sz), mode="area").reshape(C, n, sz, sz)
        px = px_mm * images.shape[-1] / sz
        new = Fusion(C, n, sz, sz, px, device=device, spall=spall, dt=dt,
                     relief=relief, depth_mm=depth_anchor_mm,
                     half_width_mm=half_width_mm, tau=tau,
                     spall_stride=spall_stride, spall_model=spall_model,
                     fix_a=fix_a, fix_b=fix_b, gain=gain)
        if model is not None:
            with torch.no_grad():
                new.a_h0.copy_(F.interpolate(model.a_h0.detach(), size=(sz, sz),
                                             mode="bicubic", align_corners=True))
                if spall_model == "levelset":
                    ds = model.spall_pot.detach()
                    new.spall_pot.copy_(F.interpolate(
                        ds, size=new.spall_pot.shape[-2:], mode="bilinear",
                        align_corners=False))
                    new.d_thresh.copy_(model.d_thresh.detach())
                    new.a_spall_depth.copy_(model.a_spall_depth.detach())
                else:
                    ds = model.d_spall.detach()
                    new.d_spall.copy_(F.interpolate(
                        ds.reshape(C * n, 1, *ds.shape[-2:]),
                        size=new.d_spall.shape[-2:], mode="bilinear",
                        align_corners=False).reshape(new.d_spall.shape))
                for k in ("a_lam", "a_eps", "a_s", "a_alpha", "d_sigma",
                          "d_kappa", "wctrl", "b", "a_rate", "b_rate", "log_g"):
                    getattr(new, k).copy_(getattr(model, k).detach())
        else:
            with torch.no_grad():
                y0 = Y[:, 0]                                   # earliest rubbing
                med = y0.reshape(C, -1).median(1).values.view(C, 1, 1)
                init = (y0 - med).clamp_min(0)
                init = init / (init.reshape(C, -1).max(1).values.view(C, 1, 1) + 1e-6)
                if relief == "levelset":
                    # level set positive where the earliest impression is white
                    new.a_h0.copy_((3.0 * (init - 0.45))[:, None])
                else:
                    init = init * depth_anchor_mm
                    new.a_h0.copy_(torch.log(torch.expm1(init.clamp_min(1e-3)))[:, None])
        model = new
        if fix_styles is not None:
            with torch.no_grad():
                inv = lambda v, lo, hi: torch.log(
                    torch.tensor(float((v - lo) / (hi - v)), device=device))
                for i, st in enumerate(fix_styles):
                    model.a_lam[i] = inv(st.rho, 0.15, 1.30)
                    model.a_eps[i] = inv(st.eps, 0.03, 0.30)
                    model.a_s[i] = inv(st.s, 0.015, 0.15)
                    model.a_alpha[i] = inv(st.alpha, 0.35, 0.99)
            for pnm in ("a_lam", "a_eps", "a_s", "a_alpha"):
                getattr(model, pnm).requires_grad_(False)
        if not warp:
            model.wctrl.requires_grad_(False)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, it)
        chunks = [torch.arange(i, min(i + chunk, C), device=device)
                  for i in range(0, C, chunk)]
        for k in range(it):
            opt.zero_grad()
            tot = dtot = 0.0
            for ci in chunks:
                wgt = len(ci) / C
                yh, _ = model(ci)
                data = (robust(yh - Y[ci], huber_c) if huber_c else
                        F.mse_loss(yh, Y[ci]))
                loss = data
                if relief == "levelset":
                    loss = loss + w_perim * tv(model.fg(ci))
                else:
                    h0 = model.h0(ci)
                    loss = loss + w_tv * tv(h0)
                    if w_dw > 0:
                        dmm = depth_anchor_mm
                        loss = loss + w_dw * ((h0 * (h0 - dmm)) ** 2).mean() / dmm ** 4
                    q = torch.quantile(h0.reshape(len(ci), -1), 0.995, dim=1)
                    loss = loss + w_anchor * ((q - depth_anchor_mm) ** 2).mean()
                if spall:
                    loss = loss + w_spall * model.spall(ci).mean()
                loss = loss + w_flow * (model.wctrl[ci] ** 2).mean()
                loss = loss + w_tone * (model.b[ci] ** 2).mean()
                if gain:
                    loss = loss + w_gain * (model.log_g ** 2).mean()
                if w_style > 0:
                    # hierarchical prior on the impression styles.  Sheets do
                    # differ, but they are all the work of one craft; left
                    # completely free, the per-sheet parameters absorb the
                    # shared weathering trend and bias the rate low.
                    lam_, eps_, s_, al_ = model.styles()
                    loss = loss + w_style * (
                        torch.log(lam_).var() + torch.log(eps_).var()
                        + torch.log(s_).var() + al_.var())
                (wgt * loss).backward()
                tot += wgt * float(loss.detach())
                dtot += wgt * float(data.detach())
            opt.step()
            sch.step()
            if verbose and (k % max(1, it // 4) == 0 or k == it - 1):
                print(f"  [{sz:>3}px] {k:5d} loss={tot:.5f} "
                      f"rmse={dtot ** 0.5:.4f}", flush=True)
    lam, eps, s, alpha = model.styles()
    with torch.no_grad():
        yh = torch.cat([model(torch.arange(i, min(i + chunk, C), device=device))[0]
                        for i in range(0, C, chunk)], 0)
    a_rate, b_rate = model.rates()
    return dict(
        a_rate=float(a_rate), b_rate=float(b_rate), final_loss=float(tot),
        final_data=float(dtot),
        model=model,
        h0=model.h0().detach()[:, 0].cpu().numpy(),
        fg=model.fg().detach()[:, 0].cpu().numpy(),
        sigma=model.sigma().detach().cpu().numpy(),
        kappa=model.kappa().detach().cpu().numpy(),
        spall=(model.spall().detach().cpu().numpy() if spall else None),
        gain=torch.exp(model.log_g).detach().cpu().numpy(),
        lam=lam.detach().cpu().numpy(), eps=eps.detach().cpu().numpy(),
        s=s.detach().cpu().numpy(), alpha=alpha.detach().cpu().numpy(),
        recon=yh.detach().cpu().numpy(),
        size=model.H,
    )
