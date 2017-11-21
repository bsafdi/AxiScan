clean:
	rm -rf build
	rm -f AxiScan/*.so

build: clean
	python setup.py build_ext --inplace

install:
	python setup.py install
