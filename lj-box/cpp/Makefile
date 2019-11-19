# This is a makefile.

PROG = simulate_many_particle_dynamics

OBJ = main.o simulationbox.o functions.o forces.o energies.o

CC = g++ -O3 -g -Wall

#LFLAG = -lgsl -lgslcblas

CFLAG = -c

OFLAG = -o

$(PROG) : $(OBJ)
	$(CC) $(OFLAG) $(PROG) $(OBJ) $(LIBS) $(LFLAG)

main.o:	simulationbox.h particle.h functions.h
interface.o: simulationbox.h functions.h
function.o: functions.h my_generate_rand.h
forces.o:
energies.o:

clean:
	rm -f *.o
