SHELL = /bin/sh
NVCC = nvcc   
COMPOPS = -Xptxas -O3 
#COMPOPS = -Xptxas -g  
#COMPOPS =   
ARCH = -arch=sm_70 
LIBS =  -lcufft -lcuda -lcudart 
INCS =  -I/usr/local/cuda-9.1/samples/common/inc   
HEADERS = include/binary.h

SRCDIR = src
OBJDIR = obj
BINDIR = bin
SRCFILES = $(SRCDIR)/main.cu
all: build

build: build_dir $(HEADERS)
	$(NVCC) $(COMPOPS) $(ARCH) $(INCS) $(SRCFILES) -o $(BINDIR)/kks.out $(LIBS)

build_dir:
	mkdir -p $(OBJDIR) 
	mkdir -p $(BINDIR)
	cp $(SRCDIR)/bin1ary bin/. 
	cp $(SRCDIR)/InputParams bin/. 

clean:
	@\rm -rf $(OBJDIR)
	@\rm -rf $(BINDIR)
	@\rm -f *.txt
	@\rm -f profile.in
#	@\rm -f job.*.*
## End of the makefile
