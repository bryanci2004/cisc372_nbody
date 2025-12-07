CC      = gcc
NVCC    = nvcc

CFLAGS   = -DDEBUG
NVCCFLAGS = -DDEBUG

LIBS    = -lm
ALWAYS_REBUILD = makefile

# Use compute_gpu.o instead of compute.o
nbody: nbody.o compute_gpu.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(CC) $(CFLAGS) -c $<

compute_gpu.o: compute_gpu.cu config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(NVCCFLAGS) -c $<

clean:
	rm -f *.o nbody
