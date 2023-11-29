from dataclasses import dataclass

@dataclass
class Configuration:
    
    #--------------------------------------------------------------------------
    # Models:
    #--------------------------------------------------------------------------    
    # 'sentence-transformers/LaBSE'
    # 'microsoft/mdeberta-v3-base' 
    # 'sentence-transformers/stsb-xlm-r-multilingual'
    # 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    # 'sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens'
    #--------------------------------------------------------------------------
    
    # Transformer
    transformer: str = 'sentence-transformers/LaBSE'
    pooling: str = 'cls'                   # 'mean' | 'cls' | 'pooler'
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    proj = None                            # None | int for lower dimension 
        
    margin: float = 0.16

    # Reduction of model size
    layers_to_keep = (0,2,4,7,9,11)                 # None -> org. model | (1,2,...,11) layers to keep
    
    # Model Destillation 
    transformer_teacher: str ='sentence-transformers/LaBSE'
    use_teacher: bool = True              # use destillation
    pooling_teacher: str = 'cls'          # 'mean' | 'cls' | 'pooler'
    proj_teacher = None                   # None | int for lower dimension 
    
    # Language Sampling
    init_pool = 0
    pool = (0,1,2,3)                      # (0,) for only train on original data without translation
    epoch_stop_switching: int = 36        # epochs no language switching more used (near end of training)
    
    # Debugging
    debug = None                          # False | 10000 for fast test
        
    # Training 
    seed: int = 42
    epochs: int = 40
    batch_size: int = 512
    mixed_precision: bool = True
    gradient_accumulation: int = 1
    gradient_checkpointing: bool = True
    verbose: bool = True
    gpu_ids: tuple = (0,1,2,3)            # GPU ids for training
    
    # Eval
    eval_every_n_epoch: int = 1
    normalize_features: bool = True
    zero_shot: bool = True                # eval before first epoch

    # Optimizer 
    clip_grad = 100.                      # None | float
    decay_exclue_bias: bool = False
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.0002                   
    scheduler: str = 'polynomial'        # 'polynomial' | 'constant' | None
    warmup_epochs: int = 2
    lr_end: float = 0.00005              #  only for 'polynomial'
    
    # Data
    language: str = 'all'                # 'all' | 'en', es', 'pt', 'fr', ....
    fold: int = 0                        # eval on fold x
    train_on_all: bool = False           # train on all data incl. data of fold x
    max_len: int = 96                    # max token lenght for topic and content
     
    # Sampling
    max_wrong: int = 128                 # limit for sampling of wrong content for specific topic
    custom_sampling: bool = True         # do custom shuffle to prevent having related content in batch
    sim_sample: bool = True              # upsample missing and combine hard negatives in batch
    sim_sample_start: int = 1            # if > 1 skip firt n epochs for sim_sampling
    
    # Save folder for model checkpoints
    model_path: str = './checkpoints_destill'
    
    # Checkpoint to start from  
    checkpoint_start = "./checkpoints/LaBSE/weights_e39_0.6660.pth"             # pre-trained checkpoint for model we want to train
    checkpoint_teacher = "./checkpoints/LaBSE/weights_e39_0.6660.pth"           # pre-trained checkpoint for maybe a teacher

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False