#! /bin/sh

nvcc -Xptxas -O3 -arch=sm_60 -I/usr/local/cuda-9.1/samples/common/inc main.cu -o kks.out -lcufft -lcuda -lcudart 
#nvcc -g -arch=sm_60 -I/usr/local/cuda-9.1/samples/common/inc main.cu -o kks.out -lcufft -lcuda -lcudart -lcurand -lcublas
