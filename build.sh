if [ ! -n "$1" ] ;then
    echo "e.g build step7"
    exit -1
fi

rm -f run_me
# g++ -O0 -g -mfma -mavx -fsanitize=address -fno-omit-frame-pointer main.cpp $1/sgemm.cpp -o run_me
# -funroll-all-loops -fomit-frame-pointer
g++ -O3 -mfma -mavx main.cpp $1/sgemm.cpp -o run_me
./run_me