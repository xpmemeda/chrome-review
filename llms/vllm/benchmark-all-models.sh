model="Baichuan2-7B-Chat"
n=32
batches=(4)
ios=("1024 16" "16 1024")
for b in "${batches[@]}"; do
    for io in "${ios[@]}"; do
        i=$(echo $io | cut -d ' ' -f 1)
        o=$(echo $io | cut -d ' ' -f 2)
        python benchmark-sync.py \
            --model /home/wnr/llms/$model \
            --num-prompts $n \
            --batch-size $b \
            --input-len $i \
            --output-len $o \
            > $model.b$b.i$i.o$o.txt 2>&1
    done
done


model="Qwen2-0.5B"
n=128
batches=(16)
ios=("1024 16" "16 1024")
for b in "${batches[@]}"; do
    for io in "${ios[@]}"; do
        i=$(echo $io | cut -d ' ' -f 1)
        o=$(echo $io | cut -d ' ' -f 2)
        python benchmark-sync.py \
            --model /home/wnr/llms/$model \
            --num-prompts $n \
            --batch-size $b \
            --input-len $i \
            --output-len $o \
            > $model.b$b.i$i.o$o.txt 2>&1
    done
done
