# Graph  based image segmentation
Based on the ["Efficient Graph-Based Image segmentation"](http://fcv2011.ulsan.ac.kr/files/announcement/413/IJCV(2004)%20Efficient%20Graph-Based%20Image%20Segmentation.pdf) paper.

Uses the "Grid Graph" model. Graph is built and managed using the networkx library. 

Uses the CIELAB "delta E" metric for edge weights instead of euclidean distances between rgb tuples or the multi-segmentation procedure described in the paper.

Example Usage:
```
python segment.py flag.jpg flag_out.jpg -k 1000 -s 0 -m 100
```
