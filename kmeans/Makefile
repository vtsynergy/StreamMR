CC := g++
CFLAGS := -c -O2 -g -I. 
LDFLAGS := -L/home/zhang/AMD-APP-SDK-v2.5-RC2-lnx64/TempSDKUtil/lib/x86_64 
LIBS := -lOpenCL -lSDKUtil
INCLUDE_DIR := -I/home/zhang/AMD-APP-SDK-v2.5-RC2-lnx64/include/ -I/home/zhang/AMD-APP-SDK-v2.5-RC2-lnx64/samples/opencl/SDKUtil/include  
APP=KMeans
SRCFILES = ${APP}.cpp ../timeRec.cpp ../StreamMR.cpp ../rdtsc.c ../scan.cpp

OBJFILES := $(patsubst %.cpp,%.o,$(SRCFILES))

EXECUTABLE := ${APP}

#all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJFILES)
	$(CC) -o $(EXECUTABLE) $(OBJFILES) $(INCLUDE_DIR) $(LDFLAGS) $(LIB_DIR) $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $<  $(INCLUDE_DIR) -o $@
	
clean:
	rm -rf *.o ../*.o $(EXECUTABLE)
