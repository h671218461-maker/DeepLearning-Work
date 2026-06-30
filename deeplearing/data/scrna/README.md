# scRNA-seq data directory

Place real single-cell RNA-seq data here when running the course project.

Recommended CSV format:

```text
cell_id,cell_type,GeneA,GeneB,GeneC,...
cell_0001,T_cell,0,12,3,...
cell_0002,B_cell,5,0,9,...
```

For a formal experiment, use a public annotated dataset such as PBMC, Tabula
Sapiens, Human Cell Atlas subsets, or a domain-specific tissue atlas. The code
also supports a synthetic smoke-test dataset through:

```bash
python train.py --synthetic-smoke-test --epochs 3
```
