
image_num=0
sh_file_path=$(dirname "$(readlink -f "$0")")
sh_file_name=$(basename "$0")
sh_file_path=$(realpath "$0")
current_time=$(date +"%m%d_%H-%M-%S")  

image_path_prefix="Kodak_dataset/"
echo "Current Korean Time: $current_time"

for image_path in "$image_path_prefix"/*; do
    ((image_num++))
    CUDA_VISIBLE_DEVICES=3 python experiment_scripts/train_img_multy.py \
        --model_type=inr_siren \
        --experiment_name SIREN_\
        --lr 0.0001 \
        --num_epochs 500 \
        --steps_til_summary 1 \
        --image_path "$image_path" \
        --loss_function image_mse_pixel_diff_loss \
        --current_time $current_time \
        --first_run $image_num \
        --sh_file_path $sh_file_path \
        --save_fig 1 \
        --debug 0 
    break
done

