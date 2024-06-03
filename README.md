# Collective Matmul Unit Tests

This repo contains a unit test to showcase the performance with and without the collective matmul tensor-parallel overlap in JAX/XLA.


## Running the tests
To run, `bash run_test.sh`. 

Ensure `--xla_gpu_threshold_for_windowed_einsum_mib=0 --xla_gpu_multi_streamed_windowed_einsum=true --xla_gpu_use_memcpy_local_p2p=true` XLA flags are set in order to use collective matmul. 

If the above flags are not used, collective matmul will not be used. In this case, you will get the results with the XLA baseline.

Use `--use_fp8` in the Python command to run this test with fp8 operators.
