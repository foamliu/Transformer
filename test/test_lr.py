import matplotlib.pyplot as plt

from config import d_model

if __name__ == '__main__':
    warmup_steps = 4000
    init_lr = d_model ** (-0.5)

    lr_list = []
    for step_num in range(1, 500000):
        # print(step_num)
        lr = init_lr * min(step_num ** (-0.65), step_num * (warmup_steps ** (-1.5)))

        lr_list.append(lr)

    plt.plot(lr_list)
    plt.show()
