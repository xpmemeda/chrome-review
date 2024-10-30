wget https://github.com/esnet/iperf/archive/refs/tags/3.20.zip
unzip 3.20.zip
cd iperf-3.20
./configure --prefix ${HOME}/local/iperf-3.20
make -j
make install
