#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import scipy.stats as st

argc = len(sys.argv)

if argc == 2:
    mean = float(sys.argv[1])
else:
    mean = 0.

data = []
for line in sys.stdin:
    data.append( float(line) )

[statistic, pvalue ] = st.ttest_1samp( data, mean )

if pvalue > 0.05:
   judge = "Accept"
else:
   judge = "Reject"

print "popmean= %f statistic= %f pvalue= %f : %s" % ( mean, statistic, pvalue, judge )

#raw_input
