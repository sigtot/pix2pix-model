import matplotlib.pyplot as plt


def plot_input_vs_output(img_in, img_out):
    classes = img_out.shape[2] if len(img_out.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img_in)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(img_out[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(img_out)
    plt.xticks([]), plt.yticks([])
    plt.show()
