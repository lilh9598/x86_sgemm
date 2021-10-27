# x86_sgemm
run: ./build.sh step7	<br/>


intel 8260	<br/>
fma peak performance 81gflop/s	<br/>

step0 naive                         : 0.607gflop/s	<br/>
step1 c code optimize               : 0.663gflop/s	<br/>
step2 kernel 8x8                    : 20.829gflop/s	<br/>
step3 Kc Mc tile                    : 21.718gflop/s	<br/>
step4 Pack B                        : 21.569gflop/s	<br/>
step5 Pack A                        : 48.245gflop/s	<br/>
step6 kernel 16x6                   : 53.913gflop/s	<br/>
step7 asm kernel16x6/aligned memory : 67.108gflop/s (82.8%)	<br/>
