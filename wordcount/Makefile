CC := g++
CFLAGS := -c -O2 -g -I. 
LDFLAGS := 
#LIBS := -lOpenCL
LIBS := -lOpenCL -lSDKUtil
LIB_DIR := -L /home/zhang/AMD-APP-SDK-v2.5-RC2-lnx64/samples/opencl/SDKUtil/build/debug/x86_64/ 
INCLUDE_DIR := -I/home/zhang/AMD-APP-SDK-v2.5-RC2-lnx64/include/ -I/home/zhang/AMD-APP-SDK-v2.5-RC2-lnx64/samples/opencl/SDKUtil/include/ 
#LIB_DIR := 
#INCLUDE_DIR := -I/home/zhang/AMD-APP-SDK-v2.5-RC2-lnx64/include/ 
SRCFILES = ../scan.cpp ../StreamMR.cpp ../ScanLargeArrays.cpp ../timeRec.c Wordcount.cpp 

OBJFILES := $(patsubst %.cpp,%.o,$(SRCFILES))

EXECUTABLE := Wordcount

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJFILES)
	$(CC) -o $@ $^ $(INCLUDE_DIR) $(LDFLAGS) $(LIB_DIR) $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $<  $(INCLUDE_DIR)  -o $@

clean:
	rm -rf *.o ../*.o $(EXECUTABLE)

