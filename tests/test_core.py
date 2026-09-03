"""Correctness tests for the forward model, the differentiable version, the
weathering operators and the transcription parser."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from scipy import ndimage as ndi

import physics as P
import weather as W

FONT = "/usr/share/fonts/truetype/arphic/ukai.ttc"
PX = 30.0 / 256


# ---------------------------------------------------------------- forward --
def test_opening_is_antiextensive_and_idempotent():
    rng = np.random.default_rng(0)
    h = np.abs(ndi.gaussian_filter(rng.standard_normal((128, 128)), 3)) * 4
    u = P.grey_open(h, 0.6, PX)
    assert (u <= h + 1e-9).all()
    assert np.allclose(P.grey_open(u, 0.6, PX), u, atol=1e-6)


def test_opening_is_increasing():
    rng = np.random.default_rng(1)
    a = np.abs(ndi.gaussian_filter(rng.standard_normal((96, 96)), 3)) * 3
    b = a + np.abs(ndi.gaussian_filter(rng.standard_normal((96, 96)), 3))
    assert (P.grey_open(a, 0.5, PX) <= P.grey_open(b, 0.5, PX) + 1e-9).all()


def test_parabolic_and_ball_elements_agree():
    m = W.glyph_mask("醴", FONT, 256)
    h = W.relief_from_mask(m, PX)
    st = P.TakingStyle(rho=0.6)
    ya = 1 - st.alpha * P.ink_coverage(P.grey_open(h, 0.6, PX, "parabolic"), st)
    yb = 1 - st.alpha * P.ink_coverage(P.grey_open(h, 0.6, PX, "ball"), st)
    assert np.abs(ya - yb).mean() < 0.02
    assert np.corrcoef(ya.ravel(), yb.ravel())[0, 1] > 0.95


def test_wide_groove_prints_white_and_hairline_is_bridged():
    h = np.zeros((160, 160))
    h[70:90, 20:140] = 1.2                       # 1.6 mm wide channel
    hair = np.zeros((160, 160))
    hair[79:81, 20:140] = 1.2                    # 0.23 mm channel
    for name in ("chanyi", "wujin", "plain"):
        st = P.STYLES[name]
        assert P.render(h, st, PX)[80, 80] > 0.95        # wide -> paper white
        assert P.render(hair, st, PX)[80, 80] < 0.6      # hairline -> inked over


def test_light_impression_keeps_more_detail_than_heavy_one():
    hair = np.zeros((160, 160))
    hair[79:81, 20:140] = 1.2
    light = P.render(hair, P.STYLES["chanyi"], PX)
    heavy = P.render(hair, P.STYLES["wujin"], PX)
    face_l = P.render(np.zeros_like(hair), P.STYLES["chanyi"], PX)[0, 0]
    face_h = P.render(np.zeros_like(hair), P.STYLES["wujin"], PX)[0, 0]
    assert (light[80, 80] - face_l) > (heavy[80, 80] - face_h)


# -------------------------------------------------------------- weathering --
def test_weathering_channels_are_monotone():
    eps = [W.Epoch(0.0, 1.0, 0.0), W.Epoch(0.2, 0.8, 0.05),
           W.Epoch(0.4, 0.6, 0.10)]
    assert W.check_monotone(eps)
    assert not W.check_monotone([eps[0], eps[2], eps[1]])


def test_spall_supports_are_nested():
    rng = np.random.default_rng(2)
    pot = W.spall_potential((128, 128), rng, PX)
    s1 = W.spall_depth(pot, 0.05, px_mm=PX) > 0.5
    s2 = W.spall_depth(pot, 0.20, px_mm=PX) > 0.5
    assert (s1 & ~s2).sum() == 0


def test_thin_strokes_are_shallower_than_thick_ones():
    m = np.zeros((128, 128), bool)
    m[40:44, 10:110] = True                       # thin
    m[70:90, 10:110] = True                       # thick
    h = W.relief_from_mask(m, PX)
    assert h[42, 60] < h[80, 60]


def test_glyph_font_covers_the_inscription():
    for c in "九成宮醴泉銘祕書監檢校侍中鉅鹿郡公臣魏徵奉勅撰":
        assert W.glyph_mask(c, FONT, 128).mean() > 0.01


# ------------------------------------------------------------------ torch --
def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    import torchmodel as T
    m = W.glyph_mask("醴", FONT, 256)
    h = W.relief_from_mask(m, PX)
    ht = torch.tensor(h, dtype=torch.float32)[None, None]
    for lam in (0.35, 0.6, 0.85):
        a = P.grey_open(h, lam, PX, "parabolic")
        b = T.grey_open_t(ht, torch.tensor([lam]), PX)[0, 0].numpy()
        assert np.abs(a - b).max() < 1e-5
    st = P.STYLES["plain"]
    ya = P.render(h, st, PX)
    yb = T.render_t(ht, *[torch.tensor([v]) for v in
                          (st.rho, st.eps, st.s, st.alpha)], PX)[0, 0].numpy()
    assert np.abs(ya - yb).max() < 1e-5


def test_edt_matches_scipy():
    torch = pytest.importorskip("torch")
    import torchmodel as T
    m = W.glyph_mask("醴", FONT, 256)
    gt = ndi.distance_transform_edt(m) * PX
    fg = torch.tensor(m.astype(np.float32))[None, None]
    d = T.edt_t(fg, PX, k=30)[0, 0].numpy()
    assert np.abs(d - gt).max() < 1e-4


def test_levelset_relief_matches_reference():
    torch = pytest.importorskip("torch")
    import torchmodel as T
    m = W.glyph_mask("醴", FONT, 256)
    ref = W.relief_from_mask(m, PX, depth_mm=1.4, half_width_mm=0.55)
    fg = torch.tensor(m.astype(np.float32))[None, None]
    h, _ = T.relief_from_levelset(fg * 30 - 15, PX, 1.4, 0.55)
    assert np.abs(h[0, 0].numpy() - ref).mean() < 0.02


# ------------------------------------------------------- transcriptions ----
def test_shiwen_parser():
    import shiwen as SW
    txt, flags = SW.parse_shiwen("維貞觀□（六）年□□（貫穿）之月")
    assert txt == "維貞觀六年貫穿之月"
    assert flags.tolist() == [0, 0, 0, 1, 0, 1, 1, 0, 0]


def test_shiwen_alignment_and_monotonicity():
    import shiwen as SW
    a = "皇帝避暑乎九成之宮"
    b = "皇帝避暑□（乎）九成之宮"
    c = "皇帝□（避）暑□（乎）九成之宮"
    ref, names, M = SW.damage_matrix({"a": a, "b": b, "c": c})
    assert ref == a
    assert M[0].sum() == 0
    assert M[1].sum() == 1 and M[2].sum() == 2
    r = SW.monotonicity(M, [0, 1, 2])
    assert r["violations"] == 0


# ------------------------------------------------------------- geometry ----
def test_lattice_recovers_a_known_pitch_and_phase():
    import segment as SG
    n, pitch, phase = 900, 37.0, 11.0
    x = np.arange(n)
    prof = 0.5 + 0.5 * np.cos(2 * np.pi * (x - phase) / pitch)
    p, theta, _ = SG.lattice_1d(prof, 20, 70)
    assert abs(p - pitch) < 1.0
    centres = np.array([p * (theta / (2 * np.pi) + m) for m in range(-2, 30)])
    err = np.min(np.abs(centres[:, None] - (phase + pitch * np.arange(20))[None, :]), 0)
    assert np.median(err) < 2.0


def test_dtw_alignment_is_monotone_and_finds_a_shift():
    import corpus as CP
    rng = np.random.default_rng(0)
    base = rng.standard_normal((40, 8))
    A = base[:35]
    B = np.vstack([rng.standard_normal((5, 8)), base[:35]])
    S = A @ B.T
    S /= np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None, :]
    pairs, _ = CP.dtw_align(S, gap=0.45)
    assert all(j - i == 5 for i, j in pairs)
    assert all(b[0] > a[0] and b[1] > a[1] for a, b in zip(pairs, pairs[1:]))
