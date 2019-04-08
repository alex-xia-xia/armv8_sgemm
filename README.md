# armv8_sgemm
This is a simple example of sgemm using assmbly code which use arm computelibrary for reference. The loop unroll factor is 2.
And we can optimize it further with register rotate skill to enlarge the loop unroll factor, but we haven't finished it.

http://blog.chinaunix.net/uid-20706279-id-1888741.html
https://blog.csdn.net/wuyao721/article/details/3573598#_Toc200964859
http://www.keil.com/support/man/docs/armclang_ref/armclang_ref_qjl1517569411293.htm


http://apfel.mathematik.uni-ulm.de/~lehn/FLENS-Trinity/flens/examples/tut01-page08.html
http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/
