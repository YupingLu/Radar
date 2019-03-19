#!/usr/bin/env python3
# Extract the first sweep from ARM CSAPR data
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 03/19/2019

# load libs
import pyart

fname = 'sgpcsaprsurI7.00.20160101.001301.raw.cfrad.20151231_230724.915_CSAP_v80_SUR.nc'

radar = pyart.io.read_cfradial(fname)
radar_zero = radar.extract_sweeps([0])

pyart.io.write_cfradial(fname+'0', radar_zero)
