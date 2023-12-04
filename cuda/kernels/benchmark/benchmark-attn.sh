b_list=(4 8 16)
s_list=(128 256 512 1024 2048)
nqnkd_list=("32 32 128" "28 4 128" "10 10 64" "14 2 64")
for b in "${b_list[@]}"; do
    for s in "${s_list[@]}"; do
        for nqnkd in "${nqnkd_list[@]}"; do
            nq=$(echo $nqnkd | cut -d ' ' -f 1)
            nk=$(echo $nqnkd | cut -d ' ' -f 2)
            d=$(echo $nqnkd | cut -d ' ' -f 3)
            python benchmark-attn.py -b $b -s $s -nq $nq -nk $nk -d $d > b${b}s${s}nq${nq}nk${nk}d${d}.txt
        done
    done
done