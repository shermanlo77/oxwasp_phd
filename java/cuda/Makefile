LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -arch=sm_75 --ptxas-options=-v --use_fast_math

all:	empiricalNullFilter

empiricalNullFilter: empiricalNullFilter.cu Makefile
	nvcc -ptx empiricalNullFilter.cu -o empiricalNullFilter.ptx $(NVCCFLAGS) $(LIB)

clean:
	rm -f empiricalNullFilter
