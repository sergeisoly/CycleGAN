# Architecture style transfer with CycleGAN

Implementation of CycleGAN paper https://arxiv.org/pdf/1703.10593.pdf
Dataset of different architecture style was taken from https://sites.google.com/site/zhexuutssjtu/projects/arch

Training colab notebook https://colab.research.google.com/drive/1TDjd2JzM4QSLeimRSMlcrc17JXWkOsyz?usp=sharing
Testing model https://colab.research.google.com/drive/1qnyEEgTdDnSzcuEAor8mjdKvRfD06NFJ#scrollTo=7Btdym_hkWbO


Some results. Generally CycleGAN can not transfer shape very well but some images look reasonable. \

Generating Gothic Style from International Style \

![Alt text](samples/X2Y_9.jpg?raw=true "Title")
![Alt text](samples/X2Y_14.jpg.jpg?raw=true "Title")

Generating International Style from Gothic \

![Alt text](samples/Y2X_10.jpg.jpg?raw=true "Title")
![Alt text](samples/Y2X_16.jpg.jpg?raw=true "Title")
