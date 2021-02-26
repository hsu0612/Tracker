import numpy as np
import cv2
x, y = np.meshgrid(np.linspace(-1,1,7), np.linspace(-1,1,7))
d = np.sqrt(x*x+y*y)
sigma, mu = 0.5, 0.0
g_kernel = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

# g_kernel2 = np.zeros(g_kernel.shape)
# g_kernel2[32:96, 32:96] = g_kernel[32:96, 32:96]
# g_kernel = g_kernel2

if __name__ == "__main__":
    # g_kernel2 = np.zeros(g_kernel.shape)
    # g_kernel2[32:96, 32:96] = g_kernel[32:96, 32:96]
    # g_kernel = g_kernel2
    print(g_kernel)
    # g_kernel = g_kernel*255
    # g_kernel = g_kernel.astype(np.uint8)
    # im_color = cv2.applyColorMap(g_kernel, cv2.COLORMAP_JET)
    # cv2.imshow("test", im_color)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
