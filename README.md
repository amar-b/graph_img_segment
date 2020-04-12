# Graph  based image segmentation
Based on the ["Efficient Graph-Based Image segmentation"](http://fcv2011.ulsan.ac.kr/files/announcement/413/IJCV(2004)%20Efficient%20Graph-Based%20Image%20Segmentation.pdf) paper.

Uses the "Grid Graph" model. Graph is built and managed using the networkx library. 

Uses the CIELAB "delta E" metric for edge weights instead of euclidean distances between rgb tuples or the multi-segmentation procedure described in the paper.

Example Usage:
```
python segment.py flag.jpg flag_out.jpg -k 1000 -s 0 -m 100
```

Help:
```
usage: segment.py [-h] [-s SIGMA] [-k K] [-m MINSIZE] input output

Graph based image segmentation

positional arguments:
  input       Input file path
  output      Output file path

optional arguments:
  -h, --help  show this help message and exit
  -s SIGMA    Sigma value for gaussian blur (default=0.5)
  -k K        Min size for each component (default=300)
  -m MINSIZE  Constant used in segmentation (default 0)
```
