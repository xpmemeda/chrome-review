mkdir deepseek-v3
cd deepseek-v3
git clone http://olafxiong:1690876163-f673dd32-1162-4584-b2db-0e5cc876288d@mmdcadamshub.polaris:16487/deepseek-ai/DeepSeek-V3.git
vllm serve DeepSeek-V3 --tensor-parallel-size 8 --quantization fp8 --max-model-len 3072 --gpu-memory-utilization 0.96 --api-key tfcc-deepseek-v3 --trust-remote-code &
sleep infinity
