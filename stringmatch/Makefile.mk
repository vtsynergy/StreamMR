#
# Copyright 2010 by Virginia Polytechnic Institute and State
# University. All rights reserved. Virginia Polytechnic Institute and
# State University (Virginia Tech) owns the software and its
# associated documentation.
#

bin_PROGRAMS += mrsm

mrsm_SOURCES = mapreduce/scan.cpp mapreduce/StreamMR.cpp  mapreduce/stringmatch/StringMatch.cpp mapreduce/timeRec.cpp
mrsm_CPPFLAGS =

all_local += mrsm-all-local
exec_local += mrsm-exec-local

mrsm-all-local:
	cp $(top_srcdir)/mapreduce/stringmatch/MapReduce_SM.cl .

mrsm-exec-local:
	cp $(top_srcdir)/mapreduce/stringmatch/MapReduce_SM.cl ${DESTDIR}${bindir}

