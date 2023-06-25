import time
from lrce.lib import *
from constants import *
from torch.utils.tensorboard import SummaryWriter
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


class AgentBase():
    def __init__(
        self,
        model: T.nn.Module,
        gpu_id: int,
        args: argparse.Namespace,
        log_enabled: bool = True,
        is_eval: bool = False,
    ) -> None:
        self.model = model
        self.args = args
        self.gpu_id = gpu_id
        self.log_enabled = log_enabled
        self.is_eval = is_eval

        self.uid = int(time.time())

        self.loss_func = T.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        if not is_eval:
            self.optim = T.optim.AdamW(
                [
                    {
                        'params': self.model.fusion_model.parameters(),
                        'lr': args.lr[0]
                    },
                    {
                        'params': self.model.text_extractor.parameters(),
                        'lr': args.lr[1]
                    },
                    {
                        'params': self.model.video_extractor.parameters(),
                        'lr': args.lr[2]
                    },
                ],
                lr=args.lr[0],
                betas=(0.9, 0.999),
            )
            self.scaler = T.cuda.amp.GradScaler()

            if args.use_cosine_scheduler:
                # self.scheduler = T.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                #     self.optim,
                #     eta_min=args.min_lr,
                #     last_epoch=-1,
                #     T_0=args.lr_restart_epoch,
                #     T_mult=args.lr_restart_mul,
                #     verbose=False,
                # )
                self.scheduler = CosineAnnealingWarmupRestarts(
                    self.optim,
                    first_cycle_steps=args.lr_restart_epoch,
                    cycle_mult=args.lr_restart_mul,
                    max_lr=args.lr[0],
                    min_lr=args.min_lr,
                    warmup_steps=args.lr_warm_up,
                    gamma=args.lr_decay_factor,
                )
            else:
                self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optim,
                    mode='max',
                    factor=args.lr_decay_factor,
                    patience=args.patience,
                    min_lr=args.min_lr,
                    verbose=False,
                )

        self.model = self.model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])

        self.logger = get_logger(__name__, gpu_id)
        if log_enabled and self.gpu_id == 0:
            self.args.log_dir = os.path.join(args.log_dir, f'{self.uid}_{args.dataset}')
            self.summary_writer = SummaryWriter(log_dir=self.args.log_dir)
            self.args.ckpt_dir = os.path.join(self.args.log_dir, 'weights')
            os.makedirs(self.args.ckpt_dir, exist_ok=True)
            self.save_config()

        self.last_loss = None
        self.last_metric_val = None
        self.counter = 0
        self.best_epoch = None
        self.best_metric_val = None

    def is_metric_val_better(self, epoch=None):
        if self.best_metric_val is None or self.last_metric_val > self.best_metric_val:
            self.best_metric_val = self.last_metric_val
            self.best_epoch = epoch
            return True
        return False

    def write_summary(self, title, value, step):
        if self.log_enabled and self.gpu_id == 0:
            self.summary_writer.add_scalar(title, value, step)

    def calculate_l2_reg(self):
        l2_reg = T.tensor(0, requires_grad=True, dtype=T.float32)
        for param in self.model.module.parameters():
            if param.requires_grad:
                l2_reg = l2_reg + param.norm(2)
        return l2_reg

    def step(
        self,
        *args: T.tensor,
        is_train: bool,
    ):
        raise NotImplementedError()

    def process_data(self, dl: T.utils.data.DataLoader, is_train: bool, epoch: int):
        if is_train:
            self.logger.info('Training Phase')
        elif not self.is_eval:
            self.logger.info('Validation Phase')

        correct_counter = T.zeros(2).to(self.gpu_id)
        batch_losses = T.zeros(len(dl)).to(self.gpu_id)

        pbar = tqdm(dl, disable=self.gpu_id != 0)

        for i, batch_data in enumerate(pbar):
            if not is_train:
                self.model.eval()
                with T.no_grad():
                    b_loss, b_total_correct, b_total_data = self.step(*batch_data, is_train)
            else:
                self.model.train()
                b_loss, b_total_correct, b_total_data = self.step(*batch_data, is_train)
                self.counter += 1

                if self.args.use_cosine_scheduler:
                    self.scheduler.step(epoch + i / len(dl))

                for k in range(len(self.optim.param_groups)):
                    self.write_summary(f'LR Scheduler/{k}', self.optim.param_groups[k]['lr'], self.counter)
                self.write_summary('Training/Batch Loss', b_loss, self.counter)
                self.write_summary('Training/Batch Accuracy', b_total_correct / b_total_data, self.counter)

                yield i

            if self.gpu_id != 0:  # reset counter for gpu rank > 0
                correct_counter[0] = 0
                correct_counter[1] = 0
            correct_counter[0] += b_total_correct
            correct_counter[1] += b_total_data
            batch_losses[i] = b_loss

            T.distributed.reduce(correct_counter, dst=0)

            avg_losses = batch_losses[batch_losses.nonzero()].mean().item()
            avg_acc = correct_counter[0] / correct_counter[1]

            pbar.set_postfix({
                'Loss': f'{avg_losses:.5f}',
                'Acc': f'{avg_acc*100:.5f}',
            })

        if not is_train:
            self.last_loss = avg_losses
            self.last_metric_val = avg_acc

            if not self.is_eval and not self.args.use_cosine_scheduler:
                self.scheduler.step(avg_acc)

            self.write_summary('Validation/Loss', avg_losses, epoch)
            self.write_summary('Validation/Accuracy', avg_acc, epoch)
        else:
            self.write_summary('Training/Loss', avg_losses, epoch)
            self.write_summary('Training/Accuracy', avg_acc, epoch)

        yield -1

    def save_config(self):
        if not self.args.debug_mode:
            del vars(self.args)['debug_mode']
        config = {**vars(self.args)}

        self.logger.info('======CONFIGURATIONS======')
        for k, v in config.items():
            self.logger.info(f'{k.upper()}: {v}')

        config_path = os.path.join(self.args.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        self.logger.info(f'Training config saved to {config_path}')

    def save_checkpoint(self, epoch: int, name: str = '', only_model: bool = True):
        if self.gpu_id == 0:
            save_checkpoint = {'model_state_dict': self.model.module.state_dict()}
            if not only_model:
                save_checkpoint['optimizer_state_dict'] = self.optim.state_dict()
                save_checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            if name != '':
                ckpt_path = os.path.join(self.args.ckpt_dir, f'{name}.pt')
            else:
                ckpt_path = os.path.join(
                    self.args.ckpt_dir,
                    f'epoch{epoch:02}_loss{self.last_loss:.4f}_metric{self.last_metric_val:.4f}.pt',
                )
            T.save(save_checkpoint, ckpt_path)
            self.logger.info(f'Checkpoint saved to {ckpt_path}')

    def load_checkpoint(self, ckpt_path: str, only_model: bool = True):
        assert os.path.exists(ckpt_path)
        checkpoint = T.load(ckpt_path)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        if not only_model:
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.logger.info(f'Succesfully loaded model in {ckpt_path}')

    def do_training(
        self,
        train_dataloader: T.utils.data.DataLoader,
        val_dataloader: T.utils.data.DataLoader,
        eval_per_epoch: int = 1,
    ):
        eval_idx = [len(train_dataloader) // eval_per_epoch * i for i in range(1, eval_per_epoch)]
        for i in range(self.args.epoch):
            self.logger.info(f'Epoch {i+1}/{self.args.epoch}')
            k = 0
            for step in self.process_data(train_dataloader, True, i):
                if step in eval_idx or step == -1:
                    deque(self.process_data(val_dataloader, False, eval_per_epoch * i + k), maxlen=0)

                    if self.is_metric_val_better(i + 1):
                        self.save_checkpoint(i + 1, 'best')
                    k += 1

            if (i + 1) % self.args.ckpt_interval == 0 or i == self.args.epoch - 1:
                self.save_checkpoint(i + 1)

            self.logger.info(f'Epoch complete\n')
        self.logger.info(f'Best result was seen in epoch {self.best_epoch}')

    def do_sanity_check(self, sanity_check_dataloader: T.utils.data.DataLoader):
        for i in range(self.args.epoch):
            self.logger.info(f'Epoch {i+1}/{self.args.epoch}')
            deque(self.process_data(sanity_check_dataloader, True, i), maxlen=0)

    def do_evaluation(self, test_dataloader: T.utils.data.DataLoader):
        deque(self.process_data(test_dataloader, False, 0), maxlen=0)
        self.logger.info(f'Accuracy: {self.last_metric_val*100:.5f}%')
        self.logger.info(f'Loss: {self.last_loss:.5f}')
