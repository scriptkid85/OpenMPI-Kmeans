TARGET = openMPI
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

dbg:
	mpirun --debug mpirun -np 1 -machinefile machinefile.local ../Debug/TestMPI -i data/input/cluster.csv -o data/res/2d.txt -k 2 -t 0.01 -l 20 -d 2

demo:
	mpirun -np 2 -machinefile machinefile.local ../Debug/TestMPI -i ../data/input/cluster.csv -o ../data/res/2d.txt -k 2 -t 0.01 -l 20 -d 2
ghc:
	mpirun -np 2 -machinefile machinefile.ghc ./TestMPI -i ../data/input/cluster.csv -o ../data/res/2d.txt -k 2 -t 0.01 -l 20 -d 2

clean:
	-rm -f *.o
	-rm -f $(TARGET)
	
