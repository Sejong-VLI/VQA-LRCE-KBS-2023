import args
import sys

from lrce.lib import *
from lrce.models.e2e import E2EOpenEnded, E2EMultipleChoice, E2ECount
from lrce.dataset.e2e_dataset import E2EMicrosoftDataset, E2ETGIFDataset
from lrce.agent import AgentOE, AgentMC, AgentCount


def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group('nccl', rank=rank, world_size=world_size)


def main(rank: int, world_size: int, eval_args: argparse.Namespace):
    ddp_setup(rank, world_size)
    T.cuda.set_device(rank)
    T.cuda.empty_cache()

    setup_logging()
    logger = get_logger(__name__, rank)

    logger.info('Preparing dataset')
    if 'tgif' in eval_args.dataset:
        tgif_type = eval_args.dataset.split('-')[-1]
        test_dataset = E2ETGIFDataset(
            split_annotation=f'{eval_args.dataset_dir}/annotations/Test_{tgif_type}_question.csv',
            full_annotation=f'{eval_args.dataset_dir}/annotations/Total_{tgif_type}_question.csv',
            videos_path=f'{eval_args.dataset_dir}/gifs',
            max_text_token_len=eval_args.text_seq_len,
            task_type=eval_args.task_type,
            sanity_check=False,
            frames_per_clip=eval_args.frame_sample_size,
            temporal_scale=eval_args.temporal_scale,
        )
    else:
        video_dict = pickle.load(open(os.path.join(eval_args.dataset_dir, 'idx-video-mapping.pkl'), 'rb'))
        test_dataset = E2EMicrosoftDataset(
            train_annotation=f'{eval_args.dataset_dir}/train_qa.json',
            val_annotation=f'{eval_args.dataset_dir}/val_qa.json',
            test_annotation=f'{eval_args.dataset_dir}/test_qa.json',
            videos_path=f'{eval_args.dataset_dir}/video',
            video_dict=video_dict,
            max_text_token_len=eval_args.text_seq_len,
            split='test',
            sanity_check=False,
            is_frame_extracted=False,
            temporal_scale=eval_args.temporal_scale,
        )

    logger.info('Instantiating model and evaluator agent')
    if eval_args.task_type == 'oe':
        model_factory = E2EOpenEnded
        agent_factory = AgentOE
    elif eval_args.task_type == 'mc':
        model_factory = E2EMultipleChoice
        agent_factory = AgentMC
    elif eval_args.task_type == 'count':
        model_factory = E2ECount
        agent_factory = AgentCount
    else:
        logger.error('Unsupported task type')
        sys.exit(-1)

    model = model_factory(
        feature_dim=eval_args.feature_dim,
        num_classes=eval_args.num_classes,
        video_feature_res=eval_args.video_feature_res,
        video_feature_dim=eval_args.video_feature_dim,
        frame_sample_size=eval_args.frame_sample_size,
        temporal_scale=eval_args.temporal_scale,
        text_seq_len=eval_args.text_seq_len,
    )

    logger.info(f'Using {torch.cuda.device_count()} GPU(s)')
    evaluator = agent_factory(model, rank, eval_args, False, True)
    evaluator.load_checkpoint(eval_args.model_path)

    logger.info('Instantiating dataloader')
    test_dataloader = T.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_args.batch_size,
        shuffle=False,
        num_workers=eval_args.num_workers,
        pin_memory=True,
        sampler=DistributedSampler(test_dataset),
    )

    evaluator.do_evaluation(test_dataloader)
    destroy_process_group()


if __name__ == '__main__':
    eval_args = args.parse_arg_eval()
    world_size = torch.cuda.device_count()
    mp.spawn(main, nprocs=world_size, args=(world_size, eval_args))
