# Paper

Springer LNCS-style write-up of the discrete-diffusion malware project.

## Files

- `main.tex`       — root document (Abstract, Intro, Related Work, Methodology, Experiments, Results, Reproducibility, Conclusion).
- `references.bib` — BibTeX bibliography (`splncs04` style).
- `figures/`       — t-SNE PNGs copied from `eval_results/<family>/` at draft time.

## Compile

### Overleaf (easiest)

1. Create a blank Overleaf project, upload everything in `paper/`.
2. Set the compiler to **pdfLaTeX**.
3. Overleaf already ships `llncs.cls` and `splncs04.bst`.

### Local TeX Live / MacTeX

```bash
# MacTeX includes llncs.cls; otherwise:
sudo tlmgr install texlive-publishers   # or your distro's equivalent
cd paper
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

If `llncs.cls not found`, install the `llncs` package via TeX Live's package
manager (`tlmgr install lncs` or similar) or grab the bundle from Springer's
author site.

## TODOs before submission

- [ ] Replace `https://github.com/<TODO>/diffusion-proj` in `main.tex` with the real URL.
- [ ] Fill in family-size cells in Table~\ref{tab:families} (only `zeroaccess` is currently quantified).
- [ ] When the multi-family D3PM run lands, replace the single-family D3PM result tables with the full sweep.
