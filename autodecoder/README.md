We additionally need [E(n)-equivariant steerable CNNs (escnn)](https://github.com/QUVA-Lab/escnn):
```
pip install escnn
```

I have tried [equivariant MLP (emlp)](https://github.com/mfinzi/equivariant-MLP) which is built on [JAX](https://github.com/google/jax) (but has a PyTorch version):
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install emlp
```
which worked but did not have batch norm implemented and its Pytorch version was slower than escnn.
