TARGET = openMPI-Kmeans
LIBS = -lm
CC = mpicc
CFLAGS = -g -Wall

.PHONY: default all clean

default: $(TARGET)

all: default

OBJECTS = ../src/*.c
HEADERS = ../src/*.h

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -Wall $(LIBS) -o $@


demo:
	mpirun -np 3 -machinefile machinefile.local ../Debug/$(TARGET) -p 0 -i ../data/input/cluster.csv -o ./2d.txt -k 2 -t 0.01 -l 20 -d 2
ghc:
	mpirun -np 3 -machinefile machinefile.ghc ./$(TARGET) -p 0 -i ../data/input/cluster.csv -o ./2d.txt -k 2 -t 0.01 -l 20 -d 2

demodna:
	mpirun -np 2 -machinefile machinefile.local ../Debug/$(TARGET) -p 1 -i ../data/input/clusterDNA.csv -o ./dna.txt -k 2 -t 0.1 -l 20 -d 10
ghcdna:
	mpirun -np 2 -machinefile machinefile.ghc ./$(TARGET) -p 1 -i ../data/input/clusterDNA.csv -o ./dna.txt -k 2 -t 0.1 -l 20 -d 10

demobig:
	mpirun -np 3 -machinefile machinefile.local ../Debug/$(TARGET) -p 0 -i ../data/input/cluster.csv -o ./2d.txt -k 2 -t 0.01 -l 20 -d 2
ghcbig:
	mpirun -np 3 -machinefile machinefile.ghc ./$(TARGET) -p 0 -i ../data/input/cluster.csv -o ./2d.txt -k 2 -t 0.01 -l 20 -d 2

demodnabig:
	mpirun -np 8 -machinefile machinefile.local ../Debug/$(TARGET) -p 1 -i ../data/input/clusterDNA.big.csv -o ./dnabig.txt -k 3 -t 0.1 -l 3000000 -d 10
ghcdnabig:
	mpirun -np 2 -machinefile machinefile.ghc ./$(TARGET) -p 1 -i ../data/input/clusterDNA.big.csv -o ./dnabig.txt -k 3 -t 0.1 -l 30000 -d 10
clean:
	-rm -f *.o
	-rm -f $(TARGET)
	
