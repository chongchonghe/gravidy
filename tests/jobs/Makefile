cpu: makejobcpu runcpu

gpu: makejobgpu rungpu

makejobcpu:
	bash ../src/maketest.bash cpu

makejobgpu:
	bash ../src/maketest.bash gpu

runcpu:
	cd ../../src; rm -f gravidy-cpu; make cpu; cd ../tests/jobs; bash ../src/runtest.bash

rungpu:
	cd ../../src; rm -f gravidy-gpu; make gpu; cd ../tests/jobs; bash ../src/runtest.bash

comparecpu:
	bash ../src/compare.bash cpu

comparegpu:
	bash ../src/compare.bash gpu
