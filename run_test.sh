export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true 
                  --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_highest_priority_async_stream=true  
                  --xla_gpu_graph_level=0 --xla_gpu_enable_triton_softmax_fusion=false 
                  --xla_gpu_enable_all_gather_combine_by_dim=true 
                  --xla_gpu_threshold_for_windowed_einsum_mib=0 --xla_gpu_multi_streamed_windowed_einsum=true --xla_gpu_use_memcpy_local_p2p=true" 

#OPTIONAL: add following lines to enable profiling
#NSYS_OUTPUT_FILE=/pax/nccl_unit_tests/profiles/llama-70b-tp-8-bf16
#NSYS_CMD="nsys profile -s none -o ${NSYS_OUTPUT_FILE} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ${NSYS_CMD} python /pax/nccl_unit_tests/unit_test_fprop.py --tp 8
