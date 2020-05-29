# Introduction

`symmys` is an in-development library for performing symmetry
detection and related tasks using tensorflow. Currently it attempts to
identify rotation transformations that leave a given point cloud
unchanged and distills these rotations into a set of n-fold symmetric
axes.

## Documentation

Browse more detailed documentation
[online](https://symmys.readthedocs.io) or build the sphinx
documentation from source:

```
git clone https://github.com/klarh/symmys
cd symmys/doc
pip install -r requirements.txt
make html
```
