#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sum = 0.0
cnt = 0
absmin = 1.e30
sigmin = 1.e30
max = -1.e30
sumpow = 0

for line in sys.stdin:
    data = float(line)
    sum += data
    sumpow += data * data
    cnt += 1
    if abs(data) < absmin:
        absmin = abs(data)
    if data < sigmin:
        sigmin = data
    if data > max:
        max = data

avr = sum/cnt
var = sumpow/cnt - avr*avr
#print sum, avr, var, sigmin, absmin, max
print "%f %f %f %f %f %f" % ( sum, avr, var, sigmin, absmin, max )

#raw_input
