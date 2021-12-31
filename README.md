## Dependency

Dependency:
- pymc3
- tqdm
- python==3.7

### Install pymc3

```bash
conda install -c conda-forge pymc3 theano-pymc mkl mkl-service
```

### Quick command
To run LDA model, please run:
```python
python lda_gensim.py # Gensim implementation
python lda_pymc3.py # PyMC3 implementation
```

To run MRF model, please run:
```python
python mrf.py
```