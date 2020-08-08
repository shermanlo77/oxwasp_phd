all:	empiricalNullFilter

empiricalNullFilter: empiricalNullFilter.cu Makefile
	nvcc -ptx empiricalNullFilter.cu -o empiricalNullFilter.ptx

clean:
	rm -f empiricalNullFilter
