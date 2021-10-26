import matplotlib.pyplot as plt

names = []

for path in names:
    # path = names[0]
    x = []
    y_train = []
    y_test = []
    ppl = []
    with open(path+"/log.txt", "r") as f:
        lines = f.readlines()[2:]
        # print(lines)
        epochs = len(lines)//2
        for i in range(epochs):
            tr = lines[i*2].strip("\n").split(" ")
            ev = lines[i*2+1].strip("\n").split(" ")
            x.append(int(tr[0]))
            y_train.append(float(tr[1]))
            y_test.append(float(ev[1]))
            ppl.append(float(ev[2]))
            # print(tr, ev)

    def draw_pic(x, y_list, labels, save_file, name):
        plt.figure()
        colors = ['blue', 'red', 'yellow']
        for i, y in enumerate(y_list):
            plt.plot(x, y, color=colors[i], label=labels[i])
        plt.legend()
        plt.title(name)
        plt.xlabel("Epoch")
        plt.ylabel("LOSS")
        plt.savefig(save_file)

    draw_pic(x, [y_train, y_test], ["train loss", "test loss"], save_file=path+"/loss.png", name="train and test Loss")
    draw_pic(x[5:], [ppl[5:]], ["ppl"], save_file=path+"/ppl.png", name="ppl")

