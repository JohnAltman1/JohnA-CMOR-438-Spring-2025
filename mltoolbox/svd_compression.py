import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class SVDImageCompressor:
    def __init__(self, k):
        """
        Initialize the compressor with the number of singular values to keep.
        :param k: Number of singular values to retain for compression.
        """
        self.k = k
        self.S = []

    def compress_channel(self, channel):
        """
        Compress a single channel using SVD.
        :param channel: 2D numpy array representing a single color channel.
        :return: Reconstructed channel after compression.
        """
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)

        self.S.append(S)

        S_k = np.diag(S[:self.k])
        U_k = U[:, :self.k]
        Vt_k = Vt[:self.k, :]
        return U_k, S_k, Vt_k

    def compress_image(self, image_path):
        """
        Compress an RGB image using SVD.
        :param image_path: Path to the input image.
        :param output_path: Path to save the compressed image.
        """
        image = Image.open(image_path)
        image_array = np.array(image)
        
        if len(image_array.shape) == 3:  # RGB image
            compressed_channels_U = []
            compressed_channels_S = []
            compressed_channels_V = []
            for i in range(3):  # Iterate over R, G, B channels
                cc_U, cc_S, cc_V,= self.compress_channel(image_array[:, :, i])
                
                compressed_channels_U.append(cc_U)
                compressed_channels_S.append(cc_S)
                compressed_channels_V.append(cc_V)
        else:  # Grayscale image
            compressed_channels_U,compressed_channels_S,compressed_channels_V = self.compress_channel(image_array).astype(np.uint8)

        return np.array(compressed_channels_U),np.array(compressed_channels_S),np.array(compressed_channels_V)
    
    def plot_sigma(self,skip=0):
        self.S = np.array(self.S)
        n = ["R","G","B"]
        if len(self.S.shape)==2:
            for i in range(3):
                plt.plot(np.sort(self.S[i])[::-1][skip:len(self.S[i])],label=n[i])

        plt.xlabel("Singular Value - Order")
        plt.ylabel("Singular Value - Value")
        plt.title("Singular Values of the Image")
        plt.legend()
        plt.show

    
def expand_image(U,S,V):
    if len(U.shape) == 3:
        I = []
        for i in range(3):
            I.append(np.dot(U[i], np.dot(S[i], V[i])))
        I = np.array(I)
        return np.stack((I[0], I[1], I[2]), axis=-1).astype(np.uint8)
    else:
        return np.dot(U, np.dot(S, V))



