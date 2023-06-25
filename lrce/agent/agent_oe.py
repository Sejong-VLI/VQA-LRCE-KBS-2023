from lrce.lib import *
from lrce.agent.agent_base import AgentBase
from constants import *
from torch.utils.tensorboard import SummaryWriter


class AgentOE(AgentBase):
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
            ce_loss = self.loss_func(out, ground_truth.to(self.gpu_id))
            loss = ce_loss + self.args.reg_strength * self.calculate_l2_reg()

        if is_train:
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        prediction = T.argmax(out, dim=1)
        total_data = prediction.shape[0]
        total_correct = T.sum(prediction == ground_truth.to(self.gpu_id)).item()

        return loss.item(), total_correct, total_data