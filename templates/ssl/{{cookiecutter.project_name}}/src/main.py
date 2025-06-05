from algorithm.train import Trainer
from model.translator import Translator
from data.data import TranslateData

import hydra
from omegaconf import DictConfig, OmegaConf

def setup(config):
    torch.manual_seed(42)
    device = config.system.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    ## SETUP ##
    device = setup()

    ## DATA ##
    data = TranslateData(config)
    print('Data Loaded.')

    ## MODEL ##
    src_vocab, tgt_vocab = len(data.vocab['de']), len(data.vocab['en'])
    model = Translator(src_vocab, tgt_vocab, config)
    print('Model Created.')

    ## ALGORITHM ##
    algorithm = Trainer(data, model, config)
    algorithm.run()
    print('Done!')

if __name__ == "__main__":
    main()
