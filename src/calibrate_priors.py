"""Calibrate the regularisation by a falsifiable criterion.

A prior is too weak if the fitted solution explains the data *better than the
truth does*: the extra freedom is then being spent on fitting noise, sheet
damage and texture, and no optimiser can be expected to return the truth.  We
therefore sweep the prior weights and report, for each setting,

    L_truth : the objective at the true relief, styles and rates, with only
              the nuisance fields (spalling, warp, tone) refitted
    L_fit   : the objective the optimiser reaches from scratch

and keep the weakest priors for which L_truth <= L_fit -- the largest model
that is still identified by the data.  This is reported in the paper as a
calibration table, not tuned against the evaluation metric.
"""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np, torch, torch.nn.functional as F
import synth as S, fuse as Fz, evaluate as E

CARVE = 632
YEARS = [1050, 1350, 1600, 1800, 2000]
CHARS = list("九成宮醴泉銘祕書監檢校侍中鉅鹿郡")[:12]


def _spall_params(mdl):
    if mdl.spall_model == "levelset":
        return [mdl.spall_pot, mdl.d_thresh, mdl.a_spall_depth]
    return [mdl.d_spall]


def truth_loss(d, px, dt, Y, C, n, size, huber, w_spall, stride, w_flow,
               iters=350, spall_model="levelset"):
    mdl = Fz.Fusion(C, n, size, size, px, dt=dt, relief="free", spall=True,
                    spall_stride=stride, spall_model=spall_model).cuda()
    with torch.no_grad():
        h0 = torch.tensor(d["h0"], device="cuda")[:, None]
        mdl.a_h0.copy_(torch.log(torch.expm1(h0.clamp_min(1e-4))))
        inv = lambda v, lo, hi: torch.log(torch.tensor(float((v - lo) / (hi - v)),
                                                       device="cuda"))
        for i, st in enumerate(d["styles"]):
            mdl.a_lam[i] = inv(st.rho, 0.15, 1.30)
            mdl.a_eps[i] = inv(st.eps, 0.03, 0.30)
            mdl.a_s[i] = inv(st.s, 0.015, 0.15)
            mdl.a_alpha[i] = inv(st.alpha, 0.35, 0.99)
        mdl.a_rate.copy_(torch.log(torch.expm1(torch.tensor(0.075))))
        mdl.b_rate.copy_(torch.log(torch.expm1(torch.tensor(0.115))))
    for p in (mdl.a_h0, mdl.a_lam, mdl.a_eps, mdl.a_s, mdl.a_alpha,
              mdl.a_rate, mdl.b_rate):
        p.requires_grad_(False)
    opt = torch.optim.Adam(_spall_params(mdl) + [mdl.wctrl, mdl.b], lr=0.05)
    tot = 0.0
    for k in range(iters):
        opt.zero_grad(); tot = 0.0
        for i in range(0, C, 4):
            ci = torch.arange(i, min(i + 4, C), device="cuda")
            yh, _ = mdl(ci)
            l = Fz.robust(yh - Y[ci], huber) \
                + w_spall * mdl.spall(ci).mean() \
                + w_flow * (mdl.wctrl[ci] ** 2).mean() \
                + 3.0 * (mdl.b[ci] ** 2).mean()
            (l * len(ci) / C).backward(); tot += float(l) * len(ci) / C
        opt.step()
    return tot


def main(out="results/prior_calibration.json", size=192):
    d = S.make_corpus(CHARS, YEARS, seed=11, size_px=size, carve_year=CARVE)
    px, m = d["px_mm"], d["masks"]
    dt = [(y - CARVE) / 100 for y in YEARS]
    Y = torch.tensor(d["images"], device="cuda"); C, n = Y.shape[:2]
    huber = 0.15
    rows = []
    for spall_model in ("levelset", "free"):
      for stride in (8, 16):
        for w_spall in (2e-3, 2e-2):
            t0 = time.time()
            Lt = truth_loss(d, px, dt, Y, C, n, size, huber, w_spall, stride,
                            1e-3, spall_model=spall_model)
            r = Fz.fit(d["images"], px, dt=dt, sizes=(96, size), iters=(500, 800),
                       verbose=False, relief="free", huber_c=huber,
                       w_spall=w_spall, spall_stride=stride,
                       spall_model=spall_model)
            mdl = r["model"]
            with torch.no_grad():
                Lf = 0.0
                for i in range(0, C, 4):
                    ci = torch.arange(i, min(i + 4, C), device="cuda")
                    yh, _ = mdl(ci)
                    l = Fz.robust(yh - Y[ci], huber) \
                        + w_spall * mdl.spall(ci).mean() \
                        + 1e-3 * (mdl.wctrl[ci] ** 2).mean() \
                        + 3.0 * (mdl.b[ci] ** 2).mean()
                    Lf += float(l) * len(ci) / C
            iou, thr = E.best_threshold(r["h0"], m, 1e-3,
                                        float(np.quantile(r["h0"], 0.999)), 40)
            row = dict(spall_model=spall_model, stride=stride, w_spall=w_spall,
                       L_truth=Lt, L_fit=Lf,
                       identified=bool(Lt <= Lf), iou=iou,
                       a=r["a_rate"], b=r["b_rate"], secs=time.time() - t0)
            rows.append(row)
            print(f"{spall_model:8s} stride={stride:3d} w_spall={w_spall:.0e}  L_truth={Lt:.5f} "
                  f"L_fit={Lf:.5f}  identified={row['identified']}  IoU={iou:.3f} "
                  f"a={r['a_rate']:.4f} ({row['secs']:.0f}s)", flush=True)
            json.dump(rows, open(out, "w"), indent=1)
    b0 = E.best_threshold(d["images"][:, 0], m, 0.3, 0.95, 40)
    print(f"baseline earliest IoU={b0[0]:.3f}")
    json.dump(dict(rows=rows, baseline_earliest=b0[0]), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
