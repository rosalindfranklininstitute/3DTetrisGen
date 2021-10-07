# 3DTetrisGen
In this project a set of 3D numpy arrays will be generated, each containing non-overlapping Tetris blocks of heterogeneous density surrounded by zero-density background.
These auto-generated arrays will subsequently be used as a simple model for testing the autoencoder developed by the stuff of CCP-EM.

Steps:
- creating representations of ~5 types of 5x5x5 Tetris blocks (first boundaries, then heterogeneous density)
- random positioning of a Tetris block inside a larger 30x30x30 array (?)
- allowing for Tetris blocks to be randomly rotated (in steps of 90 degrees) + reflected (?)
- allowing for more than one Tetris block to fit into the 30x30x30 block without the objects overlapping

Mark's ideas: 
1 seed per object --> random # of pieces (parameters of min/max) --> random connections