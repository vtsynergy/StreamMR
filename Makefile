OCLSDKROOT := /opt/OCLSDK/

CFLAGS := -c -O2 -g -I. 
LDFLAGS := -L${OCLSDKROOT}/lib/x86_64/
LIBS := -lOpenCL -lm
INCLUDE_DIR := -I${OCLSDKROOT}/include 
APP=KMeans WordCount MatrixMul StringMatch
SRCFILES = kmeans/KMeans.cpp wordcount/WordCount.cpp matrixmul/Matrixmul.cpp stringmatch/StringMatch.cpp timeRec.cpp StreamMR.cpp rdtsc.c scan.cpp

OBJFILES := timeRec.o StreamMR.o rdtsc.o scan.o

EXECUTABLE := ${APP}

all: $(EXECUTABLE)
	mkdir -p build
	cp scan.cl ./build/
	cp $(EXECUTABLE) ./build/

KMeans: kmeans/KMeans.o $(OBJFILES)
	$(CXX) -o $@ $^ $(INCLUDE_DIR) $(LDFLAGS) $(LIB_DIR) $(LIBS)
	mkdir -p build
	cp kmeans/*.cl ./build/

WordCount: wordcount/WordCount.o $(OBJFILES)
	$(CXX) -o $@ $^ $(INCLUDE_DIR) $(LDFLAGS) $(LIB_DIR) $(LIBS)
	mkdir -p build
	cp wordcount/*.cl ./build/

MatrixMul: matrixmul/Matrixmul.o $(OBJFILES)
	$(CXX) -o $@ $^ $(INCLUDE_DIR) $(LDFLAGS) $(LIB_DIR) $(LIBS)
	mkdir -p build
	cp matrixmul/*.cl ./build/

StringMatch: stringmatch/StringMatch.o $(OBJFILES)
	$(CXX) -o $@ $^ $(INCLUDE_DIR) $(LDFLAGS) $(LIB_DIR) $(LIBS)
	mkdir -p build
	cp stringmatch/*.cl ./build/

%.o: %.cpp
	$(CXX) $(CFLAGS) $<  $(INCLUDE_DIR) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $<  $(INCLUDE_DIR) -o $@

clean:
	rm -rf *.o ./matrixmul/*.o ./build/* ${APP}
