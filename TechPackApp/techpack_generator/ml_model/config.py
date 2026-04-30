# Archtecture setting
import os

class ModelConfig:

    # project root (TechPackApp/)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    #embedding dimensions 
    d_model = 256

    #number of attention heads
    num_heads = 8

    #number of encoder and decoder layers
    num_encoder_layers = 4
    num_decoder_layers = 4

    #feedforward network dimension 
    d_ff = 1024

    #dropout rate
    dropout_rate = 0.1

    #maximum sequence length
    max_seq_length = 256

    #training param

    batch_size = 16

    learning_rate = 0.0001

    num_epochs = 50 

    #50 epochs about 50,000 total steps so 1000 warmup steps
    warmup_steps = 1000

    #grad clip to prevent grqadients growing too quickly 
    grad_clip = 1.0

    #save every 5 epochs 
    save_every = 5 

    #directories

    data_dir = os.path.join(base_dir, 'techpack_generator', 'training_data')
    model_dir = os.path.join(base_dir, 'techpack_generator', 'models')
    checkpoint_dir = os.path.join(base_dir, 'techpack_generator', 'models', 'checkpoints')

    #specific files
    train_file = os.path.join(data_dir, 'techpack_training_data_train.json')
    val_file = os.path.join(data_dir, 'techpack_training_data_val.json')
    tokenizer_file = os.path.join(base_dir, 'techpack_generator', 'ml_model', 'tokenizer.json')

config = ModelConfig()


class V2Config(ModelConfig):
    # deep and narrow  more layers, smaller hidden dim, about same param count as v1
    # tests whether sequential depth beats representational width for this task
    d_model            = 192
    num_heads          = 6   # 32 per head
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff               = 768
    dropout_rate       = 0.1
    learning_rate      = 1e-4
    num_epochs         = 25
    save_every         = 5
    train_file = os.path.join(ModelConfig.data_dir, 'combined_train.json')
    val_file   = os.path.join(ModelConfig.data_dir, 'combined_val.json')


class V3Config(ModelConfig):
    # shallow and wide , fewer layers, larger hidden dim, roughly same param count as v1
    # tests whether richer per-token representations can compensate for fewer processing steps
    d_model            = 384
    num_heads          = 8   # 48 per head
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff               = 1536
    dropout_rate       = 0.1
    learning_rate      = 1e-4
    num_epochs         = 25
    save_every         = 5
    train_file = os.path.join(ModelConfig.data_dir, 'combined_train.json')
    val_file   = os.path.join(ModelConfig.data_dir, 'combined_val.json')

class V4Config(ModelConfig):
    # Parameter-Matched Asymmetric (~9.5M parameters)
    # Redistributes the baseline layer budget: 6 encoder, 2 decoder.
    # Fair test against v1 to see if shifting compute to the encoder improves garment matching.
    
    d_model            = 256     # Same width as v1 baseline
    num_heads          = 8       # 32 per head
    num_encoder_layers = 6       # Deep encoder (stolen from decoder)
    num_decoder_layers = 2       # Shallow decoder (JSON generation is easy)
    d_ff               = 1024    # Same FF width as v1 baseline
    
    dropout_rate       = 0.1
    learning_rate      = 1e-4
    num_epochs         = 25
    save_every         = 5
    
    train_file = os.path.join(ModelConfig.data_dir, 'combined_train.json')
    val_file   = os.path.join(ModelConfig.data_dir, 'combined_val.json')
