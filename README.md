# Complex PCA of spike timing

Label-free recovery of ordered hippocampal reactivations from spike-phase principal components.

```bash
conda create -n CPCA --override-channels -c conda-forge python=3.12 numpy scipy matplotlib scikit-learn pandas jupyter pytest pip
conda activate CPCA
pip install -e .
pytest -q
```

`environment.yml` lists the same spec. Use `conda create ... --override-channels -c conda-forge` rather than `conda env create -f`, which asks for Anaconda channel Terms of Service on this machine.

Full plan: `docs/complex_pca_research_plan.md`.
