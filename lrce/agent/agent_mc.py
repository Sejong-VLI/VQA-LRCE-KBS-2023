from lrce.lib import *
from lrce.agent.agent_base import AgentBase
from constants import *


class AgentMC(AgentBase):
    def __init__(
        self,
        model: T.nn.Module,
        gpu_id: int,
        args: argparse.Namespace,
        log_enabled: bool = True,
        is_eval: bool = False,
    ) -> None:
        super().__init__(model, gpu_id, args, log_enabled, is_eval)
        if self.args.use_hinge_loss:
            self.loss_func = self.hinge_loss
        self.logger = get_logger(__name__, gpu_id)

    def hinge_loss(self, out: T.tensor, gt: T.tensor) -> T.tensor:
        # out (B, TOTAL_MC)
        # gt (B) with index corresponding to correct answer
        batch, total_mc = out.shape
        loss = T.empty(batch, requires_grad=True, dtype=T.float32).to(self.gpu_id)

        # might be able to be optimized with torch gather
        for i in range(batch):
            correct_idx = gt[i].item()
            correct_prob = out[i][correct_idx]

            total_prob = T.zeros(total_mc).to(self.gpu_id)
            for j in range(total_mc):
                if j != correct_idx:
                    total_prob[j] = out[i][j] - correct_prob

            total_prob = T.concat((total_prob[:correct_idx], total_prob[correct_idx + 1:]))
            total_prob = total_prob + self.args.margin
            total_prob = T.max(total_prob, T.zeros(total_mc - 1).to(self.gpu_id))

            loss[i] = T.sum(total_prob)
        return T.mean(loss)

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
            out_loss = self.loss_func(out, ground_truth.to(self.gpu_id))
            loss = out_loss + self.args.reg_strength * self.calculate_l2_reg()

        if is_train:
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        prediction = T.argmax(out, dim=1)
        total_data = prediction.shape[0]
        total_correct = T.sum(prediction == ground_truth.to(self.gpu_id)).item()

        return loss.item(), total_correct, total_data