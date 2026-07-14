SYS_GEOS=$(ldconfig -p | awk '/libgeos.so\.[0-9]/{print $NF; exit}')
SYS_GEOS_C=$(ldconfig -p | awk '/libgeos_c.so.1/{print $NF; exit}')

export LD_PRELOAD="${SYS_GEOS}:${SYS_GEOS_C}"

#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 train_WUSU_ddp_accum_v8.py \
#  --batch-size 2  --sync_bn \
#  --amp \
#  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
#  --sched poly --sched-on-updates \

#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29710 train_WUSU_ddp_accum_v9_ForDecoder.py \
#  --accum_steps 1 --epochs 100 \
#  --batch-size 1  --sync_bn \
#  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
#  --sched poly --sched-on-updates \
#  --relation-mode pdca \

#CUDA_VISIBLE_DEVICES=7 python train_WUSU_random.py
###
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29715 train_WUSU_main_clean_pairbcd.py \
#  --use-pdca-guided-pair-decoder \
#  --local-adapter-type fadc \
#  --pdca-dend-prior-mode none \
#  --batch-size 1 \s
#  --accum_steps 1 \
#  --epochs 1 \
#  --seed 42 \
#  --sync_bn \
#  --amp \
#  --opt adamp \
#  --opt-betas 0.9 0.999 \
#  --opt-eps 1e-8 \
#  --sched poly \
#  --sched-on-updates \
#  --output_dir ./logs/ablation_A0_fadc_noprior/

#CUDA_VISIBLE_DEVICES=4,5,6,7  torchrun  --nproc_per_node=4  --master_port=29715  train_WUSU_main_clean_pairbcd.py \
#  --use-pdca-guided-pair-decoder \
#  --local-adapter-type fgs \
#  --fgs-ablation-mode uniform_route \
#  --fgs-prior-mode none \
#  --pdca-dend-prior-mode none \
#  --effective-gsd 1.0 \
#  --fgs-anneal-updates 10000 \
#  --fgs-basis-chunk-size 2 \
#  --batch-size 1 \
#  --accum_steps 1 \
#  --epochs 100 \
#  --seed 42 \
#  --sync_bn \
#  --amp \
#  --opt adamp \
#  --opt-betas 0.9 0.999 \
#  --opt-eps 1e-8 \
#  --sched poly \
#  --sched-on-updates \
#  --output_dir ./logs/ablation_A1_fgs_uniform_route/
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 \
#torchrun \
#  --nproc_per_node=4 \
#  --master_port=29715 \
#  train_WUSU_main_clean_pairbcd.py \
#  --use-pdca-guided-pair-decoder \
#  --local-adapter-type fgs \
#  --fgs-ablation-mode stats_only \
#  --fgs-prior-mode none \
#  --pdca-dend-prior-mode none \
#  --effective-gsd 1.0 \
#  --fgs-anneal-updates 10000 \
#  --fgs-basis-chunk-size 2 \
#  --batch-size 1 \
#  --accum_steps 1 \
#  --epochs 100 \
#  --seed 42 \
#  --sync_bn \
#  --amp \
#  --opt adamp \
#  --opt-betas 0.9 0.999 \
#  --opt-eps 1e-8 \
#  --sched poly \
#  --sched-on-updates \
#  --output_dir ./logs/ablation_A2_fgs_stats_only/




#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29715 train_WUSU_main_clean_pairbcd_evidence_unit.py \
#  --accum_steps 1 --epochs 100 \
#  --batch-size 1  --sync_bn \
#  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
#  --sched poly --sched-on-updates \
#  --relation-mode pdca \
#  --use-pdca-guided-pair-decoder \
#  --use-pdca-guidance true \
#  --output_dir ./logs/A2_task_evidence_no_highpass_seed3407/ \
#  --seed 3407 \
#  --task-evidence-enabled true \
#  --task-evidence-feature-residual true \
#  --task-evidence-use-highpass false \
#  --task-evidence-detach true \
#  --task-evidence-normalize true \
#  --task-evidence-use-integer-surrogate false \
#  --task-evidence-residual-init 0.001 \
#  --task-evidence-scale 3 \
#  --decoder-task-gate-enabled false \
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29715 train_WUSU_main_clean_pairbcd_evidence_unit.py \
#  --accum_steps 1 --epochs 100 \
#  --batch-size 1  --sync_bn \
#  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
#  --sched poly --sched-on-updates \
#  --relation-mode pdca \
#  --use-pdca-guided-pair-decoder \
#  --use-pdca-guidance true \
#  --output_dir ./logs/A3_task_evidence_full_seed3407/ \
#  --seed 3407 \
#  --task-evidence-enabled true \
#  --task-evidence-feature-residual true \
#  --task-evidence-use-highpass true \
#  --task-evidence-detach true \
#  --task-evidence-normalize true \
#  --task-evidence-use-integer-surrogate false \
#  --task-evidence-residual-init 0.001 \
#  --task-evidence-scale 3 \
#  --decoder-task-gate-enabled false \


# '''
# --neuron-type dend_fadc \
# --dend-spatial-conv-type structure_routed_v1 \
# --routeconv-ablation-mode MODE \
# --dend-residual-init 0.01
#         full
#     uniform_route
#     global_route
#     no_axis_descriptor
#     isotropic_direction_pool
# '''
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=25710 train_WUSU_main_clean_pairbcd.py \
  --accum_steps 1 --epochs 100 \
  --batch-size 1  --sync_bn \
  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
  --sched poly --sched-on-updates \
  --relation-mode pdca \
  --use-pdca-guided-pair-decoder \
  --output_dir ./logs/clean_train_routconvV1_uniform_route/ \
  --pretrain_from '' \
  --dend-spatial-conv-type structure_routed_v1 \
  --routeconv-ablation-mode uniform_route \
  --dend-residual-init 0.01 \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=25710 train_WUSU_main_clean_pairbcd.py \
  --accum_steps 1 --epochs 100 \
  --batch-size 1  --sync_bn \
  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
  --sched poly --sched-on-updates \
  --relation-mode pdca \
  --use-pdca-guided-pair-decoder \
  --output_dir ./logs/clean_train_routconvV1_global_route/ \
  --pretrain_from '' \
  --dend-spatial-conv-type structure_routed_v1 \
  --routeconv-ablation-mode global_route \
  --dend-residual-init 0.01

