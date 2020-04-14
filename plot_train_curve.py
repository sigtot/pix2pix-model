import re
import matplotlib.pyplot as plt
with open("training_cgan_l1", 'r') as f:
    lines = f.readlines()
    g_losses = [float(re.search('G: *(.*),', line)[1]) for line in lines]
    d_losses = [float(re.search('D: *(.*)', line)[1]) for line in lines]

    plt.subplot(121)
    plt.plot(g_losses)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(("Generator",))

    plt.subplot(122)
    plt.plot(d_losses, 'g')
    plt.xlabel("epoch")
    plt.legend(("Discriminator",))
    plt.savefig('loss.png')
    plt.show()
