SYS_GEOS=$(ldconfig -p | awk '/libgeos.so\.[0-9]/{print $NF; exit}')
SYS_GEOS_C=$(ldconfig -p | awk '/libgeos_c.so.1/{print $NF; exit}')

export LD_PRELOAD="${SYS_GEOS}:${SYS_GEOS_C}"
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train_WUSU_ddp_accum_baseline.py \
#  --batch-size 2  --sync_bn \
#  --amp \
#  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
#  --sched poly --sched-on-updates \


#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29610 train_WUSU_ddp_accum_v9_pairrel_aux_v11.py \
#  --amp \
#  --batch-size 2  --sync_bn \
#  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
#  --sched poly --sched-on-updates \
#  --relation-mode pdca --enable-pairrel-aux --pairrel-aux-weight 0.02 --pairrel-aux-warmup-epochs 5 --pairrel-aux-scales 3 \
#  --pairrel-mode unchanged_only
##--relation-mode pdca --pdca_aux

#  --relation-mode pdca --enable-pairrel-aux --pairrel-aux-weight 0.05 --pairrel-aux-warmup-epochs 5 --pairrel-aux-scales 3 \

#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29510 train_WUSU_ddp_accum_v9_ForDecoder.py \
#  --accum_steps 2 --epochs 100 \
#  --batch-size 1  --sync_bn \
#  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
#  --sched poly --sched-on-updates \
#  --relation-mode pdca \
#  --use-pdca-guided-pair-decoder
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=28710 train_WUSU_main_clean_pairbcd.py \
#  --accum_steps 1 --epochs 100 \
#  --batch-size 1  --sync_bn \
#  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
#  --sched poly --sched-on-updates \
#  --relation-mode pdca \
#  --use-pdca-guided-pair-decoder \
#  --output_dir ./logs/clean_FFTDend_encoder_walveletDendDecoder_NeuronDiv8/ \
#  --k_mode NoQlif \
#  --val-mode all_pairs \
#  CUDA_VISIBLE_DEVICES=4,5,6,7 \
#torchrun \
#  --nproc_per_node=4 \
# --master_port=29715 \
#  train_WUSU_main_clean_pairbcd.py \
#  --use-pdca-guided-pair-decoder \
#  --local-adapter-type fgs \
#  --fgs-ablation-mode joint_route \
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
#  --output_dir ./logs/ablation_A4_fgs_joint_route/
#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=28710 train_WUSU_main_clean_pairbcd_evidence_unit.py \
#  --accum_steps 1 --epochs 100 \
#  --batch-size 1  --sync_bn \
#  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
#  --sched poly --sched-on-updates \
#  --relation-mode pdca \
#  --use-pdca-guided-pair-decoder \
#  --use-pdca-guidance true \
#  --output_dir ./logs/GPU_03_A2_task_evidence_no_highpass_seed3407/ \
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
#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=28710 train_WUSU_main_clean_pairbcd_evidence_unit.py \
#  --accum_steps 1 --epochs 100 \
#  --batch-size 1  --sync_bn \
#  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
#  --sched poly --sched-on-updates \
#  --relation-mode pdca \
#  --use-pdca-guided-pair-decoder \
#  --use-pdca-guidance true \
#  --output_dir ./logs/logs/GPU_03_A3_task_evidence_full_seed3407/ \
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
#
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
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29710 train_WUSU_main_clean_pairbcd.py \
  --accum_steps 1 --epochs 100 \
  --batch-size 1  --sync_bn \
  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
  --sched poly --sched-on-updates \
  --relation-mode pdca \
  --use-pdca-guided-pair-decoder \
  --output_dir ./logs/clean_train_routconvV1_isotropic_direction_pool/ \
  --pretrain_from '' \
  --dend-spatial-conv-type structure_routed_v1 \
  --routeconv-ablation-mode isotropic_direction_pool \
  --dend-residual-init 0.01 \

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29710 train_WUSU_main_clean_pairbcd.py \
  --accum_steps 1 --epochs 100 \
  --batch-size 1  --sync_bn \
  --opt adamp --opt-betas 0.9 0.999 --opt-eps 1e-8 \
  --sched poly --sched-on-updates \
  --relation-mode pdca \
  --use-pdca-guided-pair-decoder \
  --output_dir ./logs/clean_train_routconvV1_no_axis_descriptor/ \
  --pretrain_from '' \
  --dend-spatial-conv-type structure_routed_v1 \
  --routeconv-ablation-mode no_axis_descriptor \
  --dend-residual-init 0.01
  
