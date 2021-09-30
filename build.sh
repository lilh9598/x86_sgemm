if [ ! -n "$1" ] ;then
    echo "e.g build step7"
    exit -1
fi

rm -f run_me
rm -f libsgemm.so
# g++ -O0 -g -mfma -mavx -fsanitize=address -fno-omit-frame-pointer main.cpp $1/sgemm.cpp -o run_me
# -funroll-all-loops -fomit-frame-pointer
g++ -shared -fPIC -mfma -mavx -std=c++11 -O3 -o libsgemm.so $1/*
g++ -O3 -std=c++11  -Wl,-rpath=. main.cpp libsgemm.so -o run_me
./run_me
