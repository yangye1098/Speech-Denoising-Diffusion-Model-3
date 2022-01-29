import argparse
import torch
import torchaudio
from tqdm import tqdm
#import data_loader.data_loaders as module_data
import data_loader.numpy_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

import model.diffusion as module_diffusion
import model.network as module_network


from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances

    val_dataset = config.init_obj('val_dataset', module_data)
    val_data_loader = config.init_obj('data_loader', module_data, val_dataset)
    logger.info('Finish initializing datasets')

    sample_rate = config['sample_rate']
    # build model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = config.init_obj('diffusion', module_diffusion, device=device)
    network = config.init_obj('network', module_network)
    model = config.init_obj('arch', module_arch, diffusion, network)
    # prepare model for testing
    model = model.to(device)
    model.eval()
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)



    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns), device=device)

    sample_path = config.save_dir/'samples'
    sample_path.mkdir(parents=True, exist_ok=True)

    target_path = sample_path/'target'
    output_path = sample_path/'output'
    target_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, (target, condition, name_index) in enumerate(tqdm(val_data_loader)):
            target, condition = target.to(device), condition.to(device)
            # infer from conditional input only
            output = model.infer(condition)

            #
            # save samples, or do something with output here
            #
            batch_size = condition.shape[0]
            for b in range(batch_size):
                name = val_dataset.getName(name_index[b])
                torchaudio.save(output_path/f'{name}.wav', torch.unsqueeze(output[b, :], 0).cpu(), sample_rate)
                torchaudio.save(target_path/f'{name}.wav', torch.unsqueeze(target[b, :], 0).cpu(), sample_rate)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                m = metric(output,target)
                total_metrics[i] += m * batch_size

    n_samples = len(val_data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
