#
# Copyright 2010 by Virginia Polytechnic Institute and State
# University. All rights reserved. Virginia Polytechnic Institute and
# State University (Virginia Tech) owns the software and its
# associated documentation.
#

bin_PROGRAMS += mrwc

mrwc_SOURCES = mapreduce/StreamMR.cpp  mapreduce/wordcount/WordCount.cpp mapreduce/timeRec.cpp mapreduce/scan.cpp
mrwc_CPPFLAGS =

all_local += mrwc-all-local
exec_local += mrwc-exec-local

mrwc-all-local:
	cp $(top_srcdir)/mapreduce/wordcount/MapReduce_WC.cl .
	cp $(top_srcdir)/mapreduce/wordcount/scan.cl .

mrwc-exec-local:
	cp $(top_srcdir)/mapreduce/wordcount/MapReduce_WC.cl ${DESTDIR}${bindir}
	cp $(top_srcdir)/mapreduce/wordcount/scan.cl ${DESTDIR}${bindir}

