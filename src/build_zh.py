"""Assemble the Chinese manuscript into the journal template.

The English main.tex is the skeleton: preamble, figure and table environments,
labels and bibliography are reused unchanged, so the Chinese PDF has the same
format, the same figures and the same reference list.  Only the prose is
replaced, from the mmx translations, converted from Markdown to LaTeX here.
"""
from __future__ import annotations

import re, pathlib, sys

ZH = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = pathlib.Path("paper/latex_zh")

FIG = {1: "fig:overview", 2: "fig:impressions", 3: "fig:mechanism",
       4: "fig:transcriptions", 5: "fig:validation", 6: "fig:identify",
       7: "fig:usable", 8: "fig:restoration", 9: "fig:measurement",
       10: "fig:steles"}
TAB = {1: "tab:flaking", 2: "tab:ablation", 3: "tab:restoration"}

GREEK = {"λ": r"\lambda ", "ε": r"\varepsilon ", "σ": r"\sigma ", "α": r"\alpha ",
         "θ": r"\theta ", "κ": r"\kappa ", "Φ": r"\Phi ", "μ": r"\mu "}
# symbols that may appear inside an italic run and must become LaTeX maths
OPS = {"−": "-", "≥": r"\geq ", "≤": r"\leq ", "≈": r"\approx ", "×": r"\times ",
       "±": r"\pm ", "→": r"\to ", "∗": r"\ast ", "⊖": r"\ominus ",
       "⊕": r"\oplus ", "²": "^2", "₀": "_0", "ŷ": r"\hat{y}"}
MATH = {**GREEK, **OPS}
# the same symbols in running text, outside any italic run
UNI = {"−": "-", "–": "--", "≈": r"$\approx$", "≥": r"$\geq$", "≤": r"$\leq$",
       "×": r"$\times$", "±": r"$\pm$", "→": r"$\to$", "∗": r"$\ast$",
       "⊖": r"$\ominus$", "⊕": r"$\oplus$", "²": r"$^2$", "⁻": r"$^-$",
       "⁵": r"$^5$", "⁶": r"$^6$", "₀": r"$_0$", "ŷ": r"$\hat{y}$"}


def cites(s: str) -> str:
    def rep(m):
        body = m.group(1)
        if "-" in body and "," not in body:
            a, b = body.split("-")
            keys = [f"ref{i}" for i in range(int(a), int(b) + 1)]
        else:
            keys = [f"ref{x.strip()}" for x in body.split(",")]
        return r"\citep{" + ",".join(keys) + "}"
    return re.sub(r"\^([\d,\-]+)\^", rep, s)


def refs(s: str) -> str:
    s = re.sub(r"图\s?(\d+)([a-c])", lambda m: f"图~\\ref{{{FIG[int(m.group(1))]}}}{m.group(2)}", s)
    s = re.sub(r"图\s?(\d+)", lambda m: f"图~\\ref{{{FIG[int(m.group(1))]}}}", s)
    s = re.sub(r"表\s?(\d+)", lambda m: f"表~\\ref{{{TAB[int(m.group(1))]}}}", s)
    return s


def inline_math(s: str) -> str:
    """*x* around a symbol becomes math; Greek letters outside math too."""
    ok = r"[\sA-Za-z0-9_^{}()=+\-*/.,|λεσαθκΦμ−≥≤≈×±→∗⊖⊕²₀ŷ]+"

    def em(m):
        body = m.group(1)
        if re.fullmatch(ok, body) and re.search(r"[A-Za-zλεσαθκΦμ]", body):
            for k, v in MATH.items():
                body = body.replace(k, v)
            return f"${body}$"
        return r"\emph{" + body + "}"
    s = re.sub(r"\*([^*\n]+)\*", em, s)
    for k, v in MATH.items():                      # stray Greek outside math
        s = s.replace(k, f"${v}$")
    return s


def md2tex(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    s = re.sub(r"`([^`]+)`", lambda m: r"\texttt{" + m.group(1) + "}", s)
    s = cites(s)
    s = refs(s)
    s = re.sub(r"\*\*([^*\n]+)\*\*", lambda m: r"\textbf{" + m.group(1) + "}", s)
    s = inline_math(s)
    for k, v in UNI.items():
        s = s.replace(k, v)
    s = s.replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    s = s.replace("$$", "")
    # straight quotes around Chinese text read badly; use the CJK pair
    s = re.sub(r'"([^"\n]{1,60})"', lambda m: "“" + m.group(1) + "”", s)
    return s


def sections(text: str) -> list[tuple[int, str, str]]:
    out, lvl, name, buf = [], None, None, []
    for line in text.splitlines():
        m = re.match(r"^(#{2,3})\s+(.*)$", line)
        if m:
            if name is not None:
                out.append((lvl, name, "\n".join(buf).strip()))
            lvl, name, buf = len(m.group(1)), m.group(2).strip(), []
        else:
            buf.append(line)
    if name is not None:
        out.append((lvl, name, "\n".join(buf).strip()))
    return out


def body_tex(files: list[str]) -> str:
    parts = []
    for f in files:
        for lvl, name, body in sections((ZH / f).read_text(encoding="utf-8")):
            cmd = "section" if lvl == 2 else "subsection"
            parts.append(f"\\{cmd}{{{md2tex(name)}}}\n\n{md2tex(body)}\n")
    return "\n".join(parts)


def main():
    en = (pathlib.Path("paper/latex/main.tex")).read_text(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- preamble: keep the class and packages, add Chinese fonts ---------
    pre = en[:en.index(r"\begin{document}")]
    pre = pre.replace(r"\usepackage{CJKutf8}",
                      "\\usepackage{xeCJK}\n"
                      "\\setCJKmainfont{Noto Serif CJK SC}\n"
                      "\\setCJKsansfont{Noto Sans CJK SC}\n"
                      "\\xeCJKsetup{CJKmath=true}\n"
                      "\\renewcommand{\\figurename}{图}\n"
                      "\\renewcommand{\\tablename}{表}\n"
                      "\\AtBeginDocument{\\renewcommand{\\abstractname}{摘要}%\n"
                      "  \\def\\keywordname{关键词}}")
    pre = pre.replace(r"\usepackage[T1]{fontenc}", "")

    # ---- title block -----------------------------------------------------
    zh_abs = (ZH / "01_abstract_intro.zh.txt").read_text(encoding="utf-8")
    abstract = zh_abs.split("## 摘要", 1)[1].split("---", 1)[0].strip()
    title = "拓本相隔数百年测量石刻的风化"
    head = (f"\\title[拓本测量石刻的风化]{{{title}}}\n\n"
            "\\author*[1]{\\fnm{Weishan} \\sur{Yang}}\\email{yangweishan@hbeu.edu.cn}\n"
            "\\author[2]{\\fnm{Jie} \\sur{Yang}}\\email{jieyang@hbue.edu.cn}\n\n"
            "\\affil*[1]{\\orgdiv{物理与电子信息工程学院}, \\orgname{湖北工程学院}, \\orgaddress{\\city{孝感}, \\country{中国}}}\n"
            "\\affil[2]{\\orgdiv{艺术设计学院}, \\orgname{湖北经济学院}, \\orgaddress{\\city{武汉}, \\country{中国}}}\n\n"
            f"\\abstract{{{md2tex(abstract)}}}\n\n"
            "\\keywords{石材风化, 拓本, 反问题, 计算金石学, 文化遗产}\n\n"
            "\\maketitle\n")

    # ---- prose -----------------------------------------------------------
    body = body_tex(["01_abstract_intro.zh.txt", "02_results_a.zh.txt",
                     "03_results_b.zh.txt", "04_discussion.zh.txt",
                     "05_methods_a.zh.txt", "06_methods_b.zh.txt"])
    body = body.split(r"\section{摘要}", 1)[-1]
    body = body[body.index(r"\section{引言}"):]

    # ---- backmatter: translated, in the English order --------------------
    back_zh = (ZH / "07_back.zh.txt").read_text(encoding="utf-8")
    back = []
    for lvl, name, txt in sections(back_zh):
        back.append(f"\\bmhead{{{md2tex(name)}}}\n\n{md2tex(txt)}\n")
    back = "\n".join(back)

    # ---- figures and tables: English environments, Chinese captions ------
    figs_zh = {}
    cur = None
    for line in (ZH / "08_figs.zh.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\*\*图\s?(\d+)\s*\|\s*(.*?)\*\*\s*$", line.strip())
        if m:
            cur = int(m.group(1)); figs_zh[cur] = ["\\textbf{" + md2tex(m.group(2)) + "}"]
        elif cur and line.strip():
            figs_zh[cur].append(md2tex(line.strip()))
    tail = en[en.index(r"\clearpage") if r"\clearpage" in en else len(en):]
    order = [int(re.search(r"fig(\d+)_", b).group(1))
             for b in re.findall(r"includegraphics\[width=\\textwidth\]\{(fig\d+_[a-z_]+)\}", tail)]
    for n in order:
        old = re.search(r"(\\caption\{)(?:[^{}]|\{[^{}]*\})*(\}\n\\label\{" + FIG[n] + r"\})",
                        tail, flags=re.S)
        if old and n in figs_zh:
            tail = tail[:old.start()] + "\\caption{" + " ".join(figs_zh[n]) + "}\n\\label{" + FIG[n] + "}" + tail[old.end():]
    # table captions
    tabs_zh = {}
    cur = None
    for line in (ZH / "09_tables.zh.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\*\*表\s?(\d+)\s*\|\s*(.*?)\*\*\s*(.*)$", line.strip())
        if m:
            cur = int(m.group(1))
            tabs_zh[cur] = md2tex(m.group(2) + " " + m.group(3)).strip()
    for n, lab in TAB.items():
        old = re.search(r"(\\caption\{)(?:[^{}]|\{[^{}]*\})*(\}\\label\{" + lab + r"\})", tail, flags=re.S)
        if old and n in tabs_zh:
            tail = tail[:old.start()] + "\\caption{" + tabs_zh[n] + "}\\label{" + lab + "}" + tail[old.end():]
    tail = tail.replace(r"\section*{Figures}", r"\section*{图}")
    tail = tail.replace(r"\section*{Tables}", r"\section*{表}")
    # the English file wraps accession numbers for CJKutf8; xeCJK needs none
    tail = re.sub(r"\\begin\{CJK\}\{UTF8\}\{[a-z]+\}(.*?)\\end\{CJK\}", r"\1",
                  tail, flags=re.S)

    doc = (pre + "\\begin{document}\n\n" + head + "\n" + body + "\n" + back +
           "\n\\bibliography{refs}\n\n" + tail)
    (OUT / "main_zh.tex").write_text(doc, encoding="utf-8")
    print("wrote", OUT / "main_zh.tex", len(doc), "chars")


if __name__ == "__main__":
    main()
