# SL2equivariance

# Data generation

Data generation happens in generate_data.py. The input arguments have descriptions, contained within the file (or call python generate_data.py --help). 

Example usage:

    python generate_data.py --num_train 5000 --num_val 500 --num_test 500 --input_degree 6 --mode max_det --data_dir data/equivariant/deg_6_train_5000_val_500_test_500

    python generate_data.py --num_train 5000 --num_val 500 --num_test 500 --input_degree 6 --mode min_poly --data_dir data/invariant/deg_6_train_5000_val_500_test_500

It is also possible to presave the induced matrices for SL2-elements of specified condition numbers ranges as follows. This is useful for efficient data augmentation.

    python save_induced.py --save_dir presaved --save_name induced_mats_deg_10 --num 10000 --max_degree 10 --cond_range 1 2 3 4 

# Training Different Models

Training happens via pytorch-lightining in train.py. It is necessary to direct the training script to directory where the dataset is saved, as well as to specify a directory where the model checkpoint and TensorBoard log will be saved. The input arguments have descriptions, contained within the file (or call python train.py --help). 

Example usage:

    ## SL2-Equivariant Model for Positivity Verification
    python train.py --batch_size 32 --max_epochs 200 --max_irrep 10 --num_layers 4 --num_internal_channels 50 --data_dir data/equivariant/deg_6_train_5000_val_500_test_1000 --mode max_det --save_dir trained_models_and_logs/example --normalize_data --normalize_val --no_batch_norm --invar_arch 10 10 --learning_rate 3e-4 

    ## SO2-Equivariant Model for Minimization
    python train.py --batch_size 32 --max_epochs 200 --max_irrep 10 --num_layers 3 --num_internal_channels 50 --data_dir data/invariant/deg_6_train_5000_val_500_test_1000 --mode min_poly --save_dir trained_models_and_logs/example --normalize_data --normalize_val --no_batch_norm --invar_arch 10 10 --learning_rate 3e-4 --so2_model

    ## MLP for Positivity Verification 

    python train.py --generic_model --batch_size 32 --mlp_arch 100 1000 --num_internal_channels 50 --data_dir data/equivariant/deg_6_train_5000_val_500_test_1000 --mode max_det --save_dir trained_models_and_logs/example --normalize_data --normalize_val --no_batch_norm --learning_rate 3e-4 

    ## MLP for Data Augmentation

    python train.py --generic_model --batch_size 32 --mlp_arch 100 1000 --num_internal_channels 50 --data_dir data/equivariant/deg_6_train_5000_val_500_test_1000 --mode max_det --save_dir trained_models_and_logs/example --normalize_data --normalize_val --no_batch_norm --learning_rate 3e-4 --precomputed_aug_file presaved/induced_mats_deg_10_lower_3_upper_4.pkl
