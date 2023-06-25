from lrce.lib import *
from constants import *
from lrce.agent.agent_base import AgentBase


class AgentCount(AgentBase):
    def __init__(
        self,
        model: T.nn.Module,
        gpu_id: int,
        args: argparse.Namespace,
        log_enabled: bool = True,
        is_eval: bool = False,
    ) -> None:
        super().__init__(model, gpu_id, args, log_enabled, is_eval)
        self.logger = get_logger(__name__, gpu_id)
        self.loss_func = T.nn.MSELoss(reduction='none')

    def is_metric_val_better(self, epoch=None):
        if self.best_metric_val is None or self.last_metric_val < self.best_metric_val:
            self.best_metric_val = self.last_metric_val
            self.best_epoch = epoch
            return True
        return False

    def step(
        self,
        video_clips: T.tensor,
        texts: T.tensor,
        texts_attention_mask: T.tensor,
        texts_type_ids: T.tensor,
        ground_truth: T.tensor,
        is_train: bool,
    ):
        with T.cuda.amp.autocast():
            out = self.model(
                video_clips.to(self.gpu_id),
                texts.to(self.gpu_id),
                texts_attention_mask.to(self.gpu_id),
                texts_type_ids.to(self.gpu_id),
            )
            mse_loss = self.loss_func(out, ground_truth.to(self.gpu_id))
            # print(self.gpu_id, T.mean(mse_loss), mse_loss.shape, out.shape, ground_truth.shape, mse_loss)
            loss = T.mean(mse_loss) + self.args.reg_strength * self.calculate_l2_reg()

        if is_train:
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        return loss.item(), mse_loss

    def process_data(self, dl: T.utils.data.DataLoader, is_train: bool, epoch: int):
        if is_train:
            self.logger.info('Training Phase')
        elif not self.is_eval:
            self.logger.info('Validation Phase')

        mse_counter = T.zeros(2).to(self.gpu_id)
        batch_losses = torch.zeros(len(dl)).to(self.gpu_id)

        pbar = tqdm(dl, disable=self.gpu_id != 0)

        for i, batch_data in enumerate(pbar):

            if not is_train:
                self.model.eval()
                with T.no_grad():
                    b_loss, mse_loss = self.step(*batch_data, is_train)
            else:
                self.model.train()
                b_loss, mse_loss = self.step(*batch_data, is_train)
                self.counter += 1

                if self.args.use_cosine_scheduler:
                    self.scheduler.step(epoch + i / len(dl))

                for k in range(len(self.optim.param_groups)):
                    self.write_summary(f'LR Scheduler/{k}', self.optim.param_groups[k]['lr'], self.counter)
                self.write_summary('Training/Batch Loss', b_loss, self.counter)
                self.write_summary('Training/Batch MSE', T.mean(mse_loss).item(), self.counter)

                yield i

            if self.gpu_id != 0:  # reset counter for gpu rank > 0
                mse_counter[0] = 0
                mse_counter[1] = 0
            mse_counter[0] += T.sum(mse_loss).item()
            mse_counter[1] += len(mse_loss)
            batch_losses[i] = b_loss

            T.distributed.reduce(mse_counter, dst=0)

            avg_losses = batch_losses[batch_losses.nonzero()].mean().item()
            avg_mse = mse_counter[0] / mse_counter[1]

            pbar.set_postfix({
                'Loss': f'{avg_losses:.5f}',
                'MSE': f'{avg_mse:.5f}',
            })

        if not is_train:
            self.last_loss = avg_losses
            self.last_metric_val = avg_mse

            if not self.is_eval and not self.args.use_cosine_scheduler:
                self.scheduler.step(-avg_mse)

            self.write_summary('Validation/Loss', avg_losses, epoch)
            self.write_summary('Validation/MSE', avg_mse, epoch)
        else:
            self.write_summary('Training/Loss', avg_losses, epoch)
            self.write_summary('Training/MSE', avg_mse, epoch)

        yield -1
