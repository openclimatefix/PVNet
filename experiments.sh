# -*- coding: utf-8 -*-
# This is a log of the experiments run under PVNet2.1

set -e


if false
then
    ################################################################################################
    # These have already been run.
    #
    # Note that this library has been refactored since these runs. So they will not work as they
    # are written here
    #
    # A few small changes would bes required to re-run these. For example, in the first model below
    # `pvnet.models.conv3d.encoders.DefaultPVNet2` should be replaced with
    # `pvnet.models.multimodal.encoders.encoders3d.DefaultPVNet2` and
    # `pvnet.models.conv3d.dense_networks.ResFCNet2` should be replaced with
    # `pvnet.models.multimodal.linear_networks.networks.ResFCNet2`

    ################################################################################################

    # Save pre-made batches
    cd scripts

    python save_batches.py \
        +batch_output_dir="/mnt/disks/batches2/batches_v3.1" \
        +num_train_batches=50_000 +num_val_batches=2_000 \

    cd ..

    # Set baseline to compare to
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v4.yaml \
        model.image_encoder._target_=pvnet.models.conv3d.encoders.DefaultPVNet2 \
        model.output_network._target_=pvnet.models.conv3d.dense_networks.ResFCNet2 \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.0001 \
        +model.optimizer.weight_decay=0.25 \
        +model.optimizer.amsgrad=True \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        callbacks.early_stopping.patience=10 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="pvnet2+ResFC2_slow_regx25_amsgrad_v0"

    # Use deep supervision to help break down the sources usefulness
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v4.yaml \
        model._target_=pvnet.models.conv3d.deep_supervision.Model \
        model.image_encoder._target_=pvnet.models.conv3d.encoders.DefaultPVNet2 \
        model.output_network._target_=pvnet.models.conv3d.dense_networks.ResFCNet2 \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.0001 \
        +model.optimizer.weight_decay=0.25 \
        +model.optimizer.amsgrad=True \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        callbacks.early_stopping.patience=10 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="pvnet2+ResFC2_deepsuper_slow_regx25_amsgrad_v0"

    # Was the original encoder network better/worse/same?
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v4.yaml \
        model.image_encoder._target_=pvnet.models.conv3d.encoders.DefaultPVNet \
        model.output_network._target_=pvnet.models.conv3d.dense_networks.ResFCNet2 \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.0001 \
        +model.optimizer.weight_decay=0.25 \
        +model.optimizer.amsgrad=True \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        callbacks.early_stopping.patience=10 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="pvnet+ResFC2_slow_regx25_amsgrad_v0"

    # Are we using too much regularisation?
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v4.yaml \
        model.image_encoder._target_=pvnet.models.conv3d.encoders.DefaultPVNet2 \
        model.output_network._target_=pvnet.models.conv3d.dense_networks.ResFCNet2 \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.0001 \
        +model.optimizer.weight_decay=0.01 \
        +model.optimizer.amsgrad=True \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        callbacks.early_stopping.patience=10 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="pvnet2+ResFC2_slow_regx1_amsgrad_v0"

    # Set this baseline using NWP alone
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=nwp_dwsrf_weighting.yaml \
        model.optimizer._target_=pvnet.optimizers.Adam \
        model.optimizer.lr=0.0001 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="dwsrf_weighting_slow_v3"

    # Try a different encoder network
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches/batches_v3" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v3.yaml \
        model.encoder_kwargs.model_name="efficientnet-b2" \
        +model.add_image_embedding_channel=True \
        model.output_network._target_=pvnet.models.conv3d.dense_networks.ResFCNet2 \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.0001 \
        +model.optimizer.weight_decay=0.05 \
        +model.optimizer.amsgrad=True \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        callbacks.early_stopping.patience=10 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="EffNet+ResFC2_slow_regx25_amsgrad_v1"

    # Use deep supervision and pvnet1 encoder so we can compare to pvnet2+deepsuper
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v4.yaml \
        model._target_=pvnet.models.conv3d.deep_supervision.Model \
        model.image_encoder._target_=pvnet.models.conv3d.encoders.DefaultPVNet \
        model.output_network._target_=pvnet.models.conv3d.dense_networks.ResFCNet2 \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.0001 \
        +model.optimizer.weight_decay=0.25 \
        +model.optimizer.amsgrad=True \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        callbacks.early_stopping.patience=10 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="pvnet+ResFC2_deepsuper_slow_regx25_amsgrad_v0"

    # Reset this baseline for model trained under PVNet2.0 project
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v4.yaml \
        model._target_=pvnet.models.conv3d.weather_residual.Model \
        +model.version=1 \
        model.add_image_embedding_channel=True \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.0001 \
        +model.optimizer.weight_decay=0.25 \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        +model.optimizer.amsgrad=True \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="pvnet+ResFC_weatherRes_slow_regx25_amsgrad_v1"


    # What if we exclude historical GSP as input?
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v4.yaml \
        model.image_encoder._target_=pvnet.models.conv3d.encoders.DefaultPVNet \
        model.output_network._target_=pvnet.models.conv3d.dense_networks.ResFCNet2 \
        model.include_gsp_yield_history=False \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.0001 \
        +model.optimizer.weight_decay=0.25 \
        +model.optimizer.amsgrad=True \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        callbacks.early_stopping.patience=10 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="pvnet+ResFC2_slow_regx25_amsgrad_v1_nohist"


    # How about a smaller encoder model?
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v4.yaml \
        model.image_encoder._target_=pvnet.models.conv3d.encoders.DefaultPVNet \
        model.encoder_kwargs.number_of_conv3d_layers=2 \
        model.output_network._target_=pvnet.models.conv3d.dense_networks.ResFCNet2 \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.0001 \
        +model.optimizer.weight_decay=0.25 \
        +model.optimizer.amsgrad=True \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        callbacks.early_stopping.patience=10 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="pvnet_shal+ResFC2_slow_regx25_amsgrad_v1"


    # How about a bigger outout model?
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v4.yaml \
        model.image_encoder._target_=pvnet.models.conv3d.encoders.DefaultPVNet \
        model.output_network._target_=pvnet.models.conv3d.dense_networks.ResFCNet2 \
        model.output_network_kwargs.n_res_blocks=6 \
        model.output_network_kwargs.fc_hidden_features=128 \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.0001 \
        +model.optimizer.weight_decay=0.25 \
        +model.optimizer.amsgrad=True \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        callbacks.early_stopping.patience=10 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="pvnet+ResFC2big_slow_regx25_amsgrad_v1"

    # Try the self-regularising neural network as the output network
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v4.yaml \
        model.image_encoder._target_=pvnet.models.conv3d.encoders.DefaultPVNet \
        model.output_network._target_=pvnet.models.conv3d.dense_networks.SNN \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.0001 \
        +model.optimizer.weight_decay=0.25 \
        +model.optimizer.amsgrad=True \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        callbacks.early_stopping.patience=10 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="pvnet+SNN_slow_regx25_amsgrad_v0"

    # Try using ResNet encoder
    python run.py \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.1" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v5.yaml \
        model._target_=pvnet.models.conv3d.deep_supervision.Model \
        model.image_encoder._target_=pvnet.models.conv3d.encoders.ResNet \
        model.output_network._target_=pvnet.models.conv3d.dense_networks.ResFCNet2 \
        model.sat_image_size_pixels=24 \
        model.nwp_image_size_pixels=24 \
        model.number_nwp_channels=2 \
        model.nwp_history_minutes=120 \
        model.nwp_forecast_minutes=480 \
        model.history_minutes=120 \
        model.optimizer._target_=pvnet.optimizers.AdamWReduceLROnPlateau \
        model.optimizer.lr=0.00005 \
        +model.optimizer.weight_decay=0.25 \
        +model.optimizer.amsgrad=True \
        +model.optimizer.patience=5 \
        +model.optimizer.factor=0.1 \
        +model.optimizer.threshold=0.002 \
        callbacks.early_stopping.patience=10 \
        datamodule.batch_size=4 \
        trainer.accumulate_grad_batches=32 \
        model_name="ResNet+ResFC2_deepsup_slow_regx25_amsgrad_v1"
        
        
    cd scripts

    # Re-train this model after refactoring
    python save_batches.py \
        +batch_output_dir="/mnt/disks/batches2/batches_v3.2" \
        +num_train_batches=200_000 \
        +num_val_batches=4_000

    cd ..


    python run.py \
        datamodule=premade_batches \
        datamodule.batch_dir="/mnt/disks/batches2/batches_v3.2" \
        +trainer.val_check_interval=10_000 \
        trainer.log_every_n_steps=200 \
        model=conv3d_sat_nwp_v6.yaml \
        callbacks.early_stopping.patience=20 \
        datamodule.batch_size=32 \
        trainer.accumulate_grad_batches=4 \
        model_name="pvnet+ResFC2+_slow_regx25_amsgrad_v4"


fi

cd scripts

# Changes in datapipes make this more like production
python save_batches.py \
    +batch_output_dir="/mnt/disks/batches2/batches_v3.4" \
    +num_train_batches=200_000 \
    +num_val_batches=4_000

cd ..

python run.py \
    datamodule=premade_batches \
    datamodule.batch_dir="/mnt/disks/batches2/batches_v3.4" \
    +trainer.val_check_interval=10_000 \
    trainer.log_every_n_steps=200 \
    model=multimodal.yaml \
    model.include_gsp_yield_history=False \
    +model.min_sat_delay_minutes=30 \
    callbacks.early_stopping.patience=20 \
    datamodule.batch_size=32 \
    trainer.accumulate_grad_batches=4 \
    callbacks.early_stopping.patience=10 \
    datamodule.batch_size=32 \
    trainer.accumulate_grad_batches=4 \
    model_name="pvnet+ResFC2+_slow_regx25_amsgrad_v5_nohist"