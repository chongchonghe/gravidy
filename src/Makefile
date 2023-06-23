use_boost = 0

# Makefile
# CUDA_PATH=/cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen/gcc-8.4.0/cuda-10.2.89-omyaqz6mv6t3fihltamr67fnj5xij5an
MPI_LIB_PATH=$(OPENMPI_LIBDIR)
MPI_INC_PATH=$(OPENMPI_INCDIR)

# Compilers
CXX=  g++
# NVCC= $(CUDA_PATH)/bin/nvcc
NVCC= nvcc

# FLAGS

# Boost FLAGS
BOOSTFLAGS =
ifeq ($(shell uname), Darwin)
	use_boost = 0
else
	ifeq ($(use_boost), 1)
		BOOST_LIBDIR = $(BOOST_ROOT)/lib
		BOOST_INCDIR = $(BOOST_ROOT)/include
		BOOSTFLAGS  = -L$(BOOST_LIBDIR)
		BOOSTFLAGS += -I$(BOOST_INCDIR)
		BOOSTFLAGS += -lboost_program_options
		BOOSTFLAGS += -DBOOST
	endif
endif

# Links FLAGS
LINKFLAGS  = -L$(CUDA_PATH)/lib64
LINKFLAGS += -I$(CUDA_PATH)/include
LINKFLAGS += -lcuda
LINKFLAGS += -lcudart
LINKFLAGS += -DGPU

# C++ FLAGS
CXXFLAGS  = $(BOOSTFLAGS)
#CXXFLAGS += -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 # For gcc > 5.2.1, by CCH on 2023-04-23
# CXXFLAGS += -std=c++11 # by CCH on 2023-04-23
CXXFLAGS += -O3 # Optimisation
CXXFLAGS += -Wall # Warning messages
CXXFLAGS += -fopenmp # Enable OpenMP
CXXFLAGS += -pipe # Pipes over temp files in the process
CXXFLAGS += -fstack-protector # Emit extra code to check buffer overflow
CXXFLAGS += -Wformat-security # Prevent formats vulnerabilities
CXXFLAGS += -Wpointer-arith # Prevent void size
CXXFLAGS += -Wformat-nonliteral # Prevent non literal format string
CXXFLAGS += -rdynamic # Exported symbols in dynamics symbol table
ifeq ($(shell uname), Darwin)
	CXXFLAGS += -Wl,-headerpad_max_install_names
	CXXFLAGS += -Wl,-dead_strip
else
	CXXFLAGS += -Wl,-z,relro # RO relocation table area in the final ELF
	CXXFLAGS += -Wl,-z,now # Prevent GOT overwrite attacks
	CXXFLAGS += -Wl,-O1 # Optimisation for the linker
	CXXFLAGS += -Wl,--discard-all # Discard local symbols
	CXXFLAGS += -Wl,--no-undefined # Prevent missing/unresolved symbols
endif
#CXXFLAGS += -pg # For gprof
#CXXFLAGS += -Werror # Warnings will be errors
ifeq ($(DEBUG),1)
	CXXFLAGS += -DDEBUG -g # Debugging
endif

# MPI FLAGS
MPIFLAGS  = -L$(MPI_LIB_PATH)
MPIFLAGS += -I$(MPI_INC_PATH)
MPIFLAGS += -pthread
MPIFLAGS += -lmpi
MPIFLAGS += -lmpi_cxx
MPIFLAGS += -ldl
MPIFLAGS += -D_MPI
MPIFLAGS += -DOMPI_SKIP_MPICXX

# CUDA FLAGS
NVFLAGS  = -DGPU
NVFLAGS += -pg
NVFLAGS += -O3
#NVFLAGS=-std=c++11 -arch=sm_20 # -DBOOST_NOINLINE='__attribute__ ((noinline))'
#NVFLAGS=-std=c++11 -arch=sm_20 -D_MWAITXINTRIN_H_INCLUDED # For gcc > 5.2.1

# General Objects
ifeq ($(use_boost), 1)
	OBJS  = include/utils/OptionsParser.o
else
	OBJS  = include/utils/OptionsParserBoostless.o
endif
OBJS += include/utils/NbodyUtils.o
OBJS += include/utils/Logger.o
OBJS += include/Hermite4.o
OBJS += include/NbodySystem.o

# CPU objects
OBJS_CPU  = $(OBJS)
OBJS_CPU += include/cpu/Hermite4CPU_integration.o
OBJS_CPU += include/cpu/Hermite4CPU.o

# MPI objects
OBJS_MPI  = $(OBJS)
OBJS_MPI += include/mpi/Hermite4MPI_integration.o
OBJS_MPI += include/mpi/Hermite4MPI.o
OBJS_MPI += include/mpi/MPIUtils.o

# GPU objects
OBJS_GPU  = $(OBJS)
OBJS_GPU += include/gpu/Hermite4GPU_kernels.cuo
OBJS_GPU += include/gpu/Hermite4GPU_integration.o
OBJS_GPU += include/gpu/Hermite4GPU.cuo

# Main rules
.PHONY: cpu pn mpi gpu doxy clean distclean cpus cpum mpis mpim gpus gpum

all: cpu

pn: print_pn
pn: gravidy-cpu

cpu: gravidy-cpu

mpi: CXXFLAGS+=$(MPIFLAGS)
mpi: gravidy-mpi

gpu: CXXFLAGS+=$(LINKFLAGS)
gpu: CXXFLAGS+=-lnvToolsExt
gpu: gravidy-gpu

gpurun: gpu maketest

gpuf: CXXFLAGS+=$(LINKFLAGS) -DMIXFORCE
gpuf: gravidy-gpu
gpus: CXXFLAGS+=$(LINKFLAGS) -DSINGLEFORCE
gpus: gravidy-gpu

cpuf: CXXFLAGS+= -DMIXFORCE
cpuf: gravidy-cpu
cpus: CXXFLAGS+= -DSINGLEFORCE
cpus: gravidy-cpu

mpif: CXXFLAGS+=$(MPIFLAGS) -DMIXFORCE
mpif: gravidy-mpi
mpis: CXXFLAGS+=$(MPIFLAGS) -DSINGLEFORCE
mpis: gravidy-mpi

print_pn:
	@echo "PN implementation is enabled in CPU version"

maketest:
	./make_and_run_test.sh

# Rules to generate CUDA object
%.cuo : %.cu
	$(NVCC) $(BOOSTFLAGS) $(NVFLAGS) -c $^ -o $@

# Rules to generate binaries
gravidy: gravidy.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS)

gravidy-cpu: gravidy.cpp $(OBJS_CPU)
	$(CXX) $^ -o $@ $(CXXFLAGS)

gravidy-mpi: gravidy.cpp $(OBJS_MPI)
	$(CXX) $^ -o $@ $(CXXFLAGS)

gravidy-gpu: gravidy.cpp $(OBJS_GPU)
	$(CXX) $^ -o $@ $(CXXFLAGS)

# Generate documentation
doxy:
	@doxygen -s Doxyfile

# Clean
clean:
	rm -f gravidy
	rm -f gravidy-gpu
	rm -f gravidy-cpu
	rm -f gravidy-mpi

distclean:
	rm -f include/*.o
	rm -f include/cpu/*.o
	rm -f include/utils/*.o
	rm -f include/gpu/*.{o,cuo}
	rm -f include/mpi/*.o
	rm -f gmon.out
