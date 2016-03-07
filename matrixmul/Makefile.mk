#
# Copyright 2010 by Virginia Polytechnic Institute and State
# University. All rights reserved. Virginia Polytechnic Institute and
# State University (Virginia Tech) owns the software and its
# associated documentation.
#

bin_PROGRAMS += mrmm

mrmm_SOURCES = mapreduce/scan.cpp mapreduce/StreamMR.cpp  mapreduce/matrixmul/Matrixmul.cpp mapreduce/timeRec.cpp
mrmm_CPPFLAGS =

all_local += mrmm-all-local
exec_local += mrmm-exec-local

mrmm-all-local:
	cp $(top_srcdir)/mapreduce/matrixmul/MapReduce_MM.cl .

mrmm-exec-local:
	cp $(top_srcdir)/mapreduce/matrixmul/MapReduce_MM.cl ${DESTDIR}${bindir}

