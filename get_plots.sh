# python indoorhack/tasks/plot_roc.py --dataset_type=scan --dataset_name=scannet_val --dataset_variant=10 --dataset_variant=20 --dataset_variant=30 --experiment_name=netvlad --experiment_name=innetvlad-v1-10 --experiment_name=innetvlad-v1-20 --experiment_name=innetvlad-v1-30
# python indoorhack/tasks/plot_roc.py --dataset_type=scan --dataset_name=scannet_val --dataset_variant=10 --dataset_variant=20 --dataset_variant=30 --experiment_name=netvlad --experiment_name=innetvlad-v2-10 --experiment_name=innetvlad-v2-20 --experiment_name=innetvlad-v2-30
# python indoorhack/tasks/plot_roc.py --dataset_type=scan --dataset_name=scannet_val --dataset_variant=10 --dataset_variant=20 --dataset_variant=30 --experiment_name=netvlad --experiment_name=innetvlad-v3-10 --experiment_name=innetvlad-v3-20 --experiment_name=innetvlad-v3-30
# python indoorhack/tasks/plot_roc.py --dataset_type=real_estate --dataset_name=sonar --experiment_name=netvlad --experiment_name=innetvlad-v1-10 --experiment_name=innetvlad-v1-20 --experiment_name=innetvlad-v1-30
# python indoorhack/tasks/plot_roc.py --dataset_type=real_estate --dataset_name=sonar --experiment_name=netvlad --experiment_name=innetvlad-v2-10 --experiment_name=innetvlad-v2-20 --experiment_name=innetvlad-v2-30
# python indoorhack/tasks/plot_roc.py --dataset_type=real_estate --dataset_name=sonar --experiment_name=netvlad --experiment_name=innetvlad-v3-10 --experiment_name=innetvlad-v3-20 --experiment_name=innetvlad-v3-30

# python indoorhack/tasks/plot_roc.py --dataset_type=scan --dataset_name=scannet_val \
#     --dataset_variant=10 --dataset_variant=20 --dataset_variant=30 \
#     --experiment_name=netvlad \
#     --experiment_name=innetvlad-v1-10 --experiment_name=innetvlad-v1-20 --experiment_name=innetvlad-v1-30 \
#     --experiment_name=innetvlad-v2-10 --experiment_name=innetvlad-v2-20 --experiment_name=innetvlad-v2-30 \
#     --experiment_name=innetvlad-v3-10 --experiment_name=innetvlad-v3-20 --experiment_name=innetvlad-v3-30

python indoorhack/tasks/plot_roc.py --dataset_type=scan --dataset_name=scannet_val \
    --dataset_variant=10 --dataset_variant=20 --dataset_variant=30 \
    --experiment_name=netvlad \
    --experiment_name=innetvlad-v1-30 \
    --experiment_name=innetvlad-v2-30 \
    --experiment_name=innetvlad-v3-30

python indoorhack/tasks/plot_roc.py --dataset_type=real_estate --dataset_name=sonar \
    --experiment_name=netvlad \
    --experiment_name=innetvlad-v1-30 \
    --experiment_name=innetvlad-v2-30 \
    --experiment_name=innetvlad-v3-30