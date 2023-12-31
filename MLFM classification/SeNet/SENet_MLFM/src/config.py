NUM_BLOCKS = [i for i in range(1, 17)]
BLOCK_WIDTH = [i for i in range(1, 1025) if i % 8 == 0]
BOTTLENECK_RATIO = [1, 2, 4]
GROUP_WIDTH = [1, 2, 4, 8, 16, 32]
SE_RARIO = 4
TRAIN_IMAGE_SIZE = 256
TEST_IMAGE_SIZE = 256
NUM_CLASSES = 100
EIGENVALUES = [[0.2175, 0.0188, 0.0045]]
EIGENVECTORS = [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]