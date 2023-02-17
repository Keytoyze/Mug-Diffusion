import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('params', metavar='str', nargs='+', type=str)
parser.add_argument('--ylim', type=str, default=None)
args = parser.parse_args()

def process(name, index, monitor='val/loss'):
    with open(name) as f:
        fcsv = csv.reader(f)
        head = None
        content = []

        val_loss = []
        for l in fcsv:
            if head is None:
                head = l
                print(head)
                i = head.index(index)
                monitor_i = head.index(monitor)
                # i = head.index("train/loss_epoch")
                
            else:
                content.append(l)
                if l[i] != "" and l[i] != "nan" and l[monitor_i] != 'nan':
                    # val_loss.append(float(l[7]))
                    val_loss.append(float(l[i]))
    val_loss = np.asarray(val_loss)
    return val_loss

# s1 = process("logs/2022-12-15T12-11-46_mug_diffusion/testtube/version_0/metrics.csv")
# s2 = process("logs/2022-12-15T22-00-15_mug_diffusion/testtube/version_0/metrics.csv")
# s3 = process("logs/2022-12-15T22-58-54_mug_diffusion/testtube/version_0/metrics.csv")
# s4 = process("logs/2022-12-16T17-44-32_mug_diffusion/testtube/version_0/metrics.csv")
# s5 = process("logs/2022-12-17T01-57-22_mug_diffusion/testtube/version_0/metrics.csv")

# val_loss = np.concatenate([s1, s2, s3, s4, s5])
# plt.plot([len(s1), len(s1)], [0.4, -0.1])
# plt.plot([len(val_loss), len(val_loss)], [0.4, -0.1])
# val_loss = np.concatenate([s1, s2, s3])

# val_loss_diff = val_loss[1:] - val_loss[:-1]
# plt.plot([0, len(val_loss)], [0, 0])
# plt.plot(list(range(len(val_loss_diff))), -val_loss_diff * 10)
# plt.plot(list(range(len(val_loss))), val_loss)
# # plt.ylim(0, 0.15)
# plt.savefig("temp.png")
# print(val_loss)

files = [
    "/var/chenmouxiang/mug-diffusion/logs/2023-01-09T02-49-54_mug_diffusion/testtube/version_0/metrics.csv",
    "/var/chenmouxiang/mug-diffusion/logs/2023-01-09T15-15-05_mug_diffusion/testtube/version_0/metrics.csv",
    "/var/chenmouxiang/mug-diffusion/logs/2023-01-10T11-40-10_mug_diffusion/testtube/version_0/metrics.csv",
    "/var/chenmouxiang/mug-diffusion/logs/2023-01-11T16-06-03_mug_diffusion/testtube/version_0/metrics.csv",
    "/var/chenmouxiang/mug-diffusion/logs/2023-01-14T01-53-07_mug_diffusion/testtube/version_0/metrics.csv",
    "/var/chenmouxiang/mug-diffusion/logs/2023-01-14T13-17-38_mug_diffusion/testtube/version_0/metrics.csv",
    # "/var/chenmouxiang/mug-diffusion/logs/2023-02-17T00-54-17_mug_diffusion/testtube/version_0/metrics.csv",
    "/var/chenmouxiang/mug-diffusion/logs/2023-02-17T13-37-06_mug_diffusion/testtube/version_0/metrics.csv",
]



# y = np.concatenate(list(map(lambda x: process(x, "loss_epoch"), files)))
# x = np.asarray(range(len(y)))# * int(302 * 0.8)
# plt.plot(x, y, label='train/loss_epoch')

# y = np.concatenate(list(map(lambda x: process(x, "val/precision_rice"), files)))
# x = np.asarray(range(len(y)))
# plt.plot(x, y, label='val/precision_rice')

# y = np.concatenate(list(map(lambda x: process(x, "val/recall_rice"), files)))
# x = np.asarray(range(len(y)))
# plt.plot(x, y, label='val/recall_rice')

for monitor in args.params:
    y = np.concatenate(list(map(lambda x: process(x, monitor), files)))
    x = np.asarray(range(len(y)))
    plt.plot(x, y, label=monitor)
    print(y)
plt.grid()
if args.ylim is not None:
    ylim = args.ylim.split(",")
    plt.ylim(float(ylim[0]), float(ylim[1]))
plt.legend()
plt.savefig("result.pdf")
