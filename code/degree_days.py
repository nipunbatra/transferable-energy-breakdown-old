# Obtained from http://www.weatherdatadepot.com/

dds = {}

dds[2014] = {'Austin':[x/787.0 for x in [10, 58, 72, 240, 408, 655, 715, 787, 588, 366, 43, 27]],
       'SanDiego':[x/787.0 for x in [65, 55, 136, 162,302, 262,428,433,472,374,176,69]],
       'Boulder':[x/787.0 for x in [0, 0,0,4,70, 213, 408, 314, 161, 31, 0, 0]]}

dd_keys = ['dd_'+str(x) for x in range(1,13)]
