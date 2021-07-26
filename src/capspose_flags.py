from absl import flags

# DIRECTORIES

flags.DEFINE_string('dataset_dir', "D:/Datasets", 'Dataset directory.')
flags.DEFINE_string('dataset', "panoptic_tv", 'Dataset name.')

# NETWORK PARAMETERS

# input-output
flags.DEFINE_integer('n_channels', 3, 'Number of image channels.')
flags.DEFINE_integer('n_classes', 19, 'Number of classes.')
flags.DEFINE_integer('input_width', 256, 'Images width.')
flags.DEFINE_integer('input_height', 256, 'Images height.')
flags.DEFINE_integer('input_channels', 3, 'Images channels.')
flags.DEFINE_multi_string('class_names', [''], 'Class names.')
flags.DEFINE_string('experiments_dir', "D:/Experiments",
                    'Experiments directory.')
flags.DEFINE_string('summaries_dir', "NA", 'Summaries directory.')
flags.DEFINE_string('checkpoint_dir', "NA", 'Checkpoint directory.')
flags.DEFINE_integer('step', 0, 'Training step (changed at runtime).')

# CapsNet
flags.DEFINE_multi_integer(
    'arch', [64, 8, 16, 16, 19], 'CapsNet parameters A, B, C, D, F.')
flags.DEFINE_integer('F', 5, 'CapsNet parameter F.')
flags.DEFINE_integer('K', 3, 'CapsNet parameter K.')
flags.DEFINE_integer('P', 4, 'CapsNet parameter P.')

# convolution
flags.DEFINE_integer('conv_stride', 2, 'CNN stride.')
flags.DEFINE_integer('conv_padding', 0, 'CNN padding.')
flags.DEFINE_integer('conv_kernel', 3, 'CNN kernel size.')

# caps_conv_1
flags.DEFINE_integer('caps_conv_1_stride', 2, 'ConvCaps1 stride.')
flags.DEFINE_integer('caps_conv_1_padding', 0, 'ConvCaps1 padding.')

# caps_conv_1
flags.DEFINE_integer('caps_conv_2_stride', 1, 'ConvCaps2 stride.')
flags.DEFINE_integer('caps_conv_2_padding', 0, 'ConvCaps2 padding.')

# class_caps
flags.DEFINE_integer('class_caps_stride', 1, 'ClassCaps stride.')
flags.DEFINE_integer('class_caps_padding', 0, 'ClassCaps padding.')

# TRAINING PARAMETERS

flags.DEFINE_integer('batch_size', 20, 'Batch size.')
flags.DEFINE_integer('seed', 7, 'Seed.')
flags.DEFINE_integer('n_epochs', 300, 'Number of training epochs.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0, 'Weight decay.')
flags.DEFINE_integer('routing_iter', 3, 'Number of routing iterations.')
flags.DEFINE_integer('pose_dim', 4, 'Pose matrix size.')
flags.DEFINE_integer('padding', 4, 'Paffing.')
flags.DEFINE_float('brightness', 0, 'Brightness.')
flags.DEFINE_float('contrast', 0.5, 'Contrast.')
flags.DEFINE_float('hue', 0, 'Hue.')
flags.DEFINE_float(
    'patience', 1e-3, 'Number of epochs with no improvement after which learning rate will be reduced.')
flags.DEFINE_integer('crop_dim', 32, 'Default crop size.')
flags.DEFINE_string('load_checkpoint_dir', 'NA',
                    'Load previous existing checkpoint.')
flags.DEFINE_string('mode', 'demo', 'train/test/demo.')
flags.DEFINE_boolean('test_affNIST', False, 'Test affnist.')
flags.DEFINE_boolean('resume_training', False,
                     'Resume training using a checkpoint.')
flags.DEFINE_integer('dataset_iterations', 10, 'Dataset iterations.')
flags.DEFINE_float('stddev', 0.01, 'Standard deviation.')
flags.DEFINE_integer('num_workers', 10, 'Number of workers.')
flags.DEFINE_integer('accumulation', 1, 'Gradient accumulation iterations.')
