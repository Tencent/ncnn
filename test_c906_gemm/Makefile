# target = test_fp16.cpp
# cmd = riscv64-unknown-linux-gnu-g++ test_fp16.cpp -o test_fp16 -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c906 -DC906=1 -static

# generate makefile

# makefile header

ALL= test_fp16

all: $(ALL)

test_fp16: test_fp16.cpp
	riscv64-unknown-linux-gnu-g++ test_fp16.cpp -o test_fp16 -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c906 -DC906=1 -static

clean:
	rm -rf $(ALL)
	