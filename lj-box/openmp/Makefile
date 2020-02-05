# This is a makefile.

PROG = testing_omp
OBJ = main.o simulationbox.o functions.o forces.o energies.o

CC = CC -O3 -g -Wall -fopenmp -std=c++11
LFLAG = -lm #-lgsl -lgslcblas
CFLAG = -c
OFLAG = -o

all: $(PROG)

install: 
	make all

$(PROG) : $(OBJ)
	$(CC) $(OFLAG) $(PROG) $(OBJ) $(LFLAG)
%.o : %.cpp
	$(CC) -c $(CFLAG) $< -o $@

clean:
	rm *.o

run:
	echo 0.8442 216 | time -f "running time %E \t time taken in kernel mode %S" ./testing_omp

.PHONY: all install
