import matplotlib.pyplot as plt
import os

names = [ os.path.join("expe", expename) for expename in os.listdir("expe")]
# print(names)
# exit()
for path in names:
    x = []
    y_train = []
    y_valid = []
    with open(path+"/log.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip("\n").split(" ")
            mode = tmp[0]
            if mode != "train" and mode != "valid":
                continue
            idx = int(tmp[1])
            if mode == "train":
                x.append(idx)
                y_train.append(float(tmp[2]))
            else:
                y_valid.append(float(tmp[2]))

    print(path, len(x), len(y_train), len(y_valid))
    if len(y_valid)<len(x):
        x = x[:-1]
        y_train = y_train[:-1]


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

    draw_pic(x, [y_train, y_valid], ["train loss", "valid loss"], save_file="/Users/zhaohexu/Desktop/NLP/hw3/"+path.split("/")[-1].replace("_","")+"loss.png", name="train and valid Loss")
# draw_pic(x[5:], [ppl[5:]], ["ppl"], save_file=path+"/ppl.png", name="ppl")

