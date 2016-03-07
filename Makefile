CC := g++

AMDROOTPATH := /opt/AMD-APP-SDK/

CFLAGS := -c -O2 -g -I. 
LDFLAGS := -L${AMDROOTPATH}/lib/x86_64/
LIBS := -lOpenCL
INCLUDE_DIR := -I${AMDROOTPATH}/include  
APP=KMeans WordCount MatrixMul StringMatch
SRCFILES = kmeans/KMeans.cpp wordcount/WordCount.cpp matrixmul/Matrixmul.cpp stringmatch/StringMatch.cpp timeRec.cpp StreamMR.cpp rdtsc.c scan.cpp

OBJFILES := timeRec.o StreamMR.o rdtsc.o scan.o

EXECUTABLE := ${APP}

all: $(EXECUTABLE)
	mkdir -p build
	cp scan.cl build
	cp $(EXECUTABLE) build 

KMeans: kmeans/KMeans.o $(OBJFILES)
	$(CC) -o $@ $^ $(INCLUDE_DIR) $(LDFLAGS) $(LIB_DIR) $(LIBS)
	cp kmeans/*.cl build

WordCount: wordcount/WordCount.o $(OBJFILES)
	$(CC) -o $@ $^ $(INCLUDE_DIR) $(LDFLAGS) $(LIB_DIR) $(LIBS)
	cp wordcount/*.cl build

MatrixMul: matrixmul/Matrixmul.o $(OBJFILES)
	$(CC) -o $@ $^ $(INCLUDE_DIR) $(LDFLAGS) $(LIB_DIR) $(LIBS)
	cp matrixmul/*.cl build

StringMatch: stringmatch/StringMatch.o $(OBJFILES)
	$(CC) -o $@ $^ $(INCLUDE_DIR) $(LDFLAGS) $(LIB_DIR) $(LIBS)
	cp stringmatch/*.cl build

%.o: %.cpp
	$(CC) $(CFLAGS) $<  $(INCLUDE_DIR) -o $@
	
clean:
	rm -rf *.o  build/* ${APP}
