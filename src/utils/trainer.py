import os
import torch
import transformers
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from dataclasses import field, dataclass
from transformers import Trainer, TrainingArguments, TrainerCallback, DefaultFlowCallback, ProgressCallback
from torch.utils.data import DataLoader
from .typings import *

from transformers.training_args import (
    cached_property,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    torch_required,
    get_int_from_env
)

from transformers.trainer import (
    math,
    time,
    sys,
    version,
    ShardedDDPOption,
    DebugOption,
    TrainerState,
    HPSearchBackend,
    RandomSampler,
    DistributedSampler,
    IterableDatasetShard,
    TrainOutput,
    deepspeed_init,
    has_length,
    hp_params,
    speed_metrics,
    dep_version_check,
    is_apex_available,
    is_fairscale_available,
    TRAINER_STATE_NAME,
    logger
)

if is_apex_available():
    from apex import amp

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    smp.init()

# prevent warning of transformers
transformers.logging.set_verbosity_error()


def train(model, loaders):
    """
    train the model

    Args:
        model
        loaders: returned by :func:`~utils.manager.Manager.prepare`
    """
    from .data import prepare_train_data, default_collate

    config = model.config
    train_dataset = prepare_train_data(config)

    args = AdonTrainingArguments(
        do_train=True,
        output_dir=model.ckpt_dir,
        # disable any report_to callbacks
        report_to="none",
        # keep all values output from the dataset
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # keep the progress callback
        disable_tqdm=False,
        logging_nan_inf_filter=False,
        # align trainingarguments.device to our config.device
        device_index=config.device,
        per_device_train_batch_size=config.batch_size,
        seed=config.seed,
        num_train_epochs=config.epoch,
        dataloader_num_workers=config.num_worker,
        max_steps=config.get("max_step", 0),
        fp16=config.fp16,
        gradient_accumulation_steps=config.accumulate_step,
        eval_delay=config.eval_delay,
        eval_steps=config.eval_step,
        save_at_eval=config.save_at_eval,
        learning_rate=config.learning_rate,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.scheduler,
        warmup_ratio=config.warmup_ratio,
        warmup_steps=config.warmup_steps,
        metric_for_best_model=config.main_metric
    )

    trainer = AdonTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=default_collate,
        loaders=loaders
    )
    trainer.remove_callback(DefaultFlowCallback)
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(AdonFlowCallback)
    trainer.add_callback(AdonProgressCallback)
    trainer.train()


@dataclass
class AdonTrainingArguments(TrainingArguments):
    device_index: DEVICE = field(
        default=0,
        metadata={"help": "The device to put model and data."}
    )
    eval_steps: Union[str, int] = field(default=0, metadata={"help": "Run an evaluation every X steps."})
    eval_delay: Union[str, int] = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed"
            )
        }
    )
    save_at_eval: bool = field(default=False, metadata={"help": "Save model at each evaluation time?"})

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
            self.local_rank = get_int_from_env(
                ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"],
                self.local_rank,
            )
            if self.local_rank != -1 and not torch.distributed.is_initialized():
                # Initializes distributed backend for cpu
                if self.xpu_backend not in ("mpi", "ccl"):
                    raise ValueError(
                        "CPU distributed training backend is not properly set. "
                        "Please set '--xpu_backend' to either 'mpi' or 'ccl'."
                    )
                if self.xpu_backend == "ccl" and int(os.environ.get("CCL_WORKER_COUNT", 0)) < 1:
                    raise ValueError(
                        "CPU distributed training backend is ccl. but CCL_WORKER_COUNT is not correctly set. "
                        "Please use like 'export CCL_WORKER_COUNT = 1' to set."
                    )

                # Try to get launch configuration from environment variables set by MPI launcher - works for Intel MPI, OpenMPI and MVAPICH
                rank = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0)
                size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1)
                local_size = get_int_from_env(
                    ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
                )
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(size)
                os.environ["LOCAL_RANK"] = str(self.local_rank)
                if not os.environ.get("MASTER_PORT", None):
                    os.environ["MASTER_PORT"] = "29500"
                if not os.environ.get("MASTER_ADDR", None):
                    if local_size != size or self.xpu_backend != "mpi":
                        raise ValueError(
                            "Looks like distributed multinode run but MASTER_ADDR env not set, "
                            "please try exporting rank 0's hostname as MASTER_ADDR"
                        )
                torch.distributed.init_process_group(backend=self.xpu_backend, rank=rank, world_size=size)
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif is_sagemaker_mp_enabled():
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            self._n_gpu = 1
        elif is_sagemaker_dp_enabled():
            import smdistributed.dataparallel.torch.torch_smddp  # noqa: F401

            dist.init_process_group(backend="smddp")
            self.local_rank = int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.deepspeed:
            # deepspeed inits torch.distributed internally
            from .deepspeed import is_deepspeed_available

            if not is_deepspeed_available():
                raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
            import deepspeed

            deepspeed.init_distributed()

            # workaround for setups like notebooks where the launcher can't be used,
            # but deepspeed requires a dist env.
            # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
            self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.local_rank == -1:
            # modified: just use the args.device as the default device
            # set _n_gpu to 1 to avoid Trainer activating DataParallel
            device = torch.device(self.device_index)
            self._n_gpu = 1
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device



class AdonFlowCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        """
        Handles when to evaluate the model.
        """
        control.should_log = False
        control.should_save = False
        control.should_evaluate = False

        if isinstance(args.eval_steps, str):
            args.eval_steps = round(state.num_update_steps_per_epoch * float(args.eval_steps[:-1]))
        if isinstance(args.eval_delay, str):
            args.eval_delay = round(state.num_update_steps_per_epoch * float(args.eval_delay[:-1]))

        if args.eval_steps > 0 and state.global_step % args.eval_steps == 0:
            # must wait until the eval_delay
            if state.global_step > args.eval_delay:
                control.should_evaluate = True

        return control



class AdonProgressCallback(TrainerCallback):
    def __init__(self):
        self.training_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(total=state.max_steps, ncols=100)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            # :.4f limits 4 decimal places to display; :.3E uses scitific notation with 3 decimal places
            self.training_bar.set_description(f"Epoch={state.epoch:.3f}, Loss={state.tr_loss:.4f}, LR={state.learning_rate:.3E}")
            self.current_step = state.global_step



class AdonTrainer(Trainer):
    def __init__(self, loaders, *args, **kwargs):
        """
        Override the huggingface Trainer

        Args:
            loaders: the loaders generated from manager.prepare; used in evaluating
        """
        super().__init__(*args, **kwargs)
        self.loaders = loaders

    def create_optimizer(self):
        optimizer = self.model.create_optimizer()
        if optimizer is None:
            super().create_optimizer()
        else:
            self.optimizer = optimizer
        return self.optimizer

    def _prepare_inputs(self, inputs):
        """
        Disable trainer to convert data to specific device.
        """
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(inputs)
        return (loss, None) if return_outputs else loss

    def evaluate(self, **kwargs):
        if self.state.is_world_process_zero:
            # neat printing
            print()

        # release cache
        self._memory_tracker.start()
        metrics = self.model.evaluate(self.loaders, log=False)

        if self.state.is_world_process_zero:
            metrics["step"] = self.state.global_step
            metrics["_best"] = metrics[self.args.metric_for_best_model]
            # the first evaluation time, the model has no main metric stored
            if "_best" not in self.model.metrics:
                self.model.metrics["_best"] = -1

            if metrics["_best"] > self.model.metrics["_best"]:
                # update metrics
                self.model.metrics["_best"] = metrics["_best"]
                if self.args.save_at_eval:
                    self.model.save(self.state.global_step)
                else:
                    self.model.save()
                # save to log file
                self.model.log_result()

        # very important to set it back; otherwise the evaluate function may be called twice at epoch end
        self.control.should_evaluate = False
        # release cache
        self._memory_tracker.stop_and_update_metrics()


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Optimization steps per epoch = {num_update_steps_per_epoch}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        # CUSTOMIZED: save how many steps are there in one epoch
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if version.parse(torch.__version__) < version.parse("1.11") or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    # modify here, add tr_loss and learning_rate to the callback so that it can be printed on the tqdm progress bar
                    tqdm_log_loss = tr_loss.item() / (self.state.global_step - self._globalstep_last_logged)
                    tqdm_log_lr = self._get_learning_rate()
                    # set to state so that it can be accessed in callbacks
                    self.state.tr_loss = tqdm_log_loss
                    self.state.learning_rate = tqdm_log_lr
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        logger.info(f"Best Metrics: {self.model.metrics}")

        return TrainOutput(self.state.global_step, train_loss, metrics)

