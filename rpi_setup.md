## Compile python3.10 from source 

Install required libraries
```bash
sudo apt install libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.6 libgdm-dev libdb4o-cil-dev libpcap-dev
```

- get the tar.gz from https://www.python.org/downloads/source/
- extract it
- cd into the extracted folder
- configure
```bash
./configure --enable-optimizations --with-openssl=/usr/bin/openssl --with-openssl-rpath=/usr/include/openssl/
```
- make 
```bash
sudo make -j 1
```
