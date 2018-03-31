import sys
import matplotlib
import matplotlib.pyplot as plt
import re


font = {'family' : 'normal',
        'size'   : 15}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.color_cycle'] = ['r', 'b', 'g']

raw_data_loss = """
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -14041561.412106
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -22689613.342159
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -25171443.291375
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -31417681.372819
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -22684451.344750
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -28371027.509497
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -19384569.665623
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -25589477.472935
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -17070627.162396
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -24111435.850717
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -15376325.296737
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -23226056.910831
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -40774070.195447
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -43665983.774668
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -33809905.118557
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -37223200.200410
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -34141547.321955
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -35813477.861683
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -35327581.167812
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -36539333.561266
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -36814496.815330
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -38446856.297847
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -38867384.115556
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -41245631.140832
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -23930429.443405
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -31062211.303430
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -20736285.122460
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -27615445.026728
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -16514659.878059
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -24586074.512452
"""

raw_data_acc = """
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 21.64 % Semantic accuracy: 24.92 % Syntactic accuracy: 19.30 %
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 30.66 % Semantic accuracy: 41.05 % Syntactic accuracy: 23.25 %
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 31.31 % Semantic accuracy: 35.46 % Syntactic accuracy: 28.35 %
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 5.90 % Semantic accuracy: 6.05 % Syntactic accuracy: 5.79 %
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 32.59 % Semantic accuracy: 38.03 % Syntactic accuracy: 28.71 %
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 17.55 % Semantic accuracy: 20.54 % Syntactic accuracy: 15.42 %
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 30.47 % Semantic accuracy: 37.85 % Syntactic accuracy: 25.20 %
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 27.57 % Semantic accuracy: 36.06 % Syntactic accuracy: 21.53 %
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 27.42 % Semantic accuracy: 33.47 % Syntactic accuracy: 23.11 %
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 31.68 % Semantic accuracy: 43.92 % Syntactic accuracy: 22.97 %
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 23.88 % Semantic accuracy: 28.16 % Syntactic accuracy: 20.83 %
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 30.94 % Semantic accuracy: 41.22 % Syntactic accuracy: 23.62 %
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 7.80 % Semantic accuracy: 7.34 % Syntactic accuracy: 8.13 %
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 8.35 % Semantic accuracy: 9.55 % Syntactic accuracy: 7.50 %
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 17.84 % Semantic accuracy: 19.44 % Syntactic accuracy: 16.70 %
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 1.49 % Semantic accuracy: 2.22 % Syntactic accuracy: 0.96 %
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 18.01 % Semantic accuracy: 21.45 % Syntactic accuracy: 15.56 %
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 6.74 % Semantic accuracy: 7.69 % Syntactic accuracy: 6.06 %
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 14.69 % Semantic accuracy: 17.44 % Syntactic accuracy: 12.73 %
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 9.41 % Semantic accuracy: 11.74 % Syntactic accuracy: 7.75 %
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 11.80 % Semantic accuracy: 13.24 % Syntactic accuracy: 10.78 %
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 11.16 % Semantic accuracy: 14.62 % Syntactic accuracy: 8.69 %
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 10.51 % Semantic accuracy: 10.84 % Syntactic accuracy: 10.28 %
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 9.42 % Semantic accuracy: 11.73 % Syntactic accuracy: 7.78 %
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 26.54 % Semantic accuracy: 28.44 % Syntactic accuracy: 25.19 %
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 6.58 % Semantic accuracy: 7.28 % Syntactic accuracy: 6.08 %
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 26.50 % Semantic accuracy: 27.54 % Syntactic accuracy: 25.77 %
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 20.63 % Semantic accuracy: 24.12 % Syntactic accuracy: 18.13 %
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 23.66 % Semantic accuracy: 27.60 % Syntactic accuracy: 20.85 %
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 29.23 % Semantic accuracy: 38.08 % Syntactic accuracy: 22.92 %
"""

def extract_name_fields(name):
    matches = re.match("FINAL_vectors_datasettext8_epochs([0-9]+)_size([0-9]+)_neg([0-9]+)_window10_sample1e-4_Q([0-9]+)_mincount5.bin_evaluated_output",
                       name)
    epochs, size, q = matches.group(1), matches.group(2), matches.group(4)
    return (int(epochs), int(size), int(q))

def extract_data(raw_data, value_field=2):
    d = []
    for line in raw_data.split("\n"):
        if line == "":
            continue
        vals = line.split()
        name, accuracy = extract_name_fields(vals[0]), float(vals[value_field])
        d.append((name, accuracy))
    return d

def plot_accuracy_vs_dimension(points_loss, points_acc, n_epochs_ran, keepqs=None):

    # Group together points of the same quantization
    unique_qs = set([d[0][2] for d in points_loss])
    unique_qs = [x for x in unique_qs if x in keepqs]
    grouped_by_qs_acc = []
    for q in unique_qs:
        grouped_by_qs_acc.append([d for d in points_acc if d[0][2] == q])
    grouped_by_qs_loss = []
    for q in unique_qs:
        grouped_by_qs_loss.append([d for d in points_loss if d[0][2] == q])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for data_acc, data_loss in zip(grouped_by_qs_acc, grouped_by_qs_loss):
        q = data_acc[0][0][2]
        q = 32 if q == 0 else q
        points_acc = [(d[0][1], d[1]) for d in data_acc]
        points_acc = sorted(points_acc, key=lambda x: x[0])
        xs = [d[0] for d in points_acc]
        ys = [d[1] for d in points_acc]
        points_loss = [(d[0][1], d[1]) for d in data_loss]
        points_loss = sorted(points_loss, key=lambda x: x[0])
        ys_loss = [d[1] for d in points_loss]
        ax1.plot(xs, ys, label="bits=" + str(q) + " (acc)", marker="o", linewidth=5, markersize=10)
        ax2.plot(xs, ys_loss, label="bits=" + str(q) + " (loss)", linestyle=":", marker="o", linewidth=5, markersize=10)

    # Merge legend names
    handles,labels = [],[]
    for ax in fig.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    plt.legend(handles,labels, loc="best")

    ax1.grid()
    #ax1.set_title("Accuracy/Loss vs Dimension, 100MB of Wikipedia, %d epochs trained" % n_epochs_ran)
    ax1.set_xlabel("Vector Dimension")
    ax2.set_ylabel("Training Loss")
    ax1.set_ylabel("Google Analogy Accuracy %")
    fig.tight_layout()
    fig.savefig("Wiki8AccuracyVsDimensionEpochsTrained=%d.pdf" % n_epochs_ran)

def plot_accuracy_vs_epochs(points_loss, points_acc, dimension, keepqs=None):

    # Group together points of the same quantization
    unique_qs = set([d[0][2] for d in points_loss])
    unique_qs = [x for x in unique_qs if x in keepqs]
    grouped_by_qs_acc = []
    for q in unique_qs:
        grouped_by_qs_acc.append([d for d in points_acc if d[0][2] == q])
    grouped_by_qs_loss = []
    for q in unique_qs:
        grouped_by_qs_loss.append([d for d in points_loss if d[0][2] == q])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for data_acc, data_loss in zip(grouped_by_qs_acc, grouped_by_qs_loss):
        q = data_acc[0][0][2]
        q = 32 if q == 0 else q
        points_acc = [(d[0][0], d[1]) for d in data_acc]
        points_acc = sorted(points_acc, key=lambda x: x[0])
        xs = [d[0] for d in points_acc]
        ys = [d[1] for d in points_acc]
        points_loss = [(d[0][0], d[1]) for d in data_loss]
        points_loss = sorted(points_loss, key=lambda x: x[0])
        ys_loss = [d[1] for d in points_loss]
        ax1.plot(xs, ys, label="Bits=" + str(q) + " (acc)", marker="o", linewidth=5, markersize=10)
        ax2.plot(xs, ys_loss, label="Bits=" + str(q) + " (loss)", linestyle=":", marker="o", linewidth=5, markersize=10)

    # Merge legend names
    handles,labels = [],[]
    for ax in fig.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    plt.legend(handles,labels, loc="best")

    ax1.grid()
    #ax1.set_title("Accuracy/Loss vs Epochs, 100MB of Wikipedia, Dimension %d" % dimension)
    ax1.set_xlabel("Epochs Trained")
    ax1.set_ylabel("Google Analogy Accuracy %")
    ax2.set_ylabel("Training Loss")
    fig.tight_layout()
    fig.savefig("Wiki8AccuracyVsEpochsTrainedDimension=%d.pdf" % dimension)


data_losses = extract_data(raw_data_loss)
# Losses are in negative form, so negate
data_losses = [(x[0], -x[1]) for x in data_losses]
data_accs = extract_data(raw_data_acc, value_field=3)

assert(set([x[0] for x in data_losses]) == set([x[0] for x in data_accs]))

# Plot (Accuracy vs Dimension including lines for each Q) for each epochs
unique_epochs = set([d[0][0] for d in data_losses])
for epoch in unique_epochs:
    datapoints_loss = [d for d in data_losses if d[0][0] == epoch]
    datapoints_accs = [d for d in data_accs if d[0][0] == epoch]
    plot_accuracy_vs_dimension(datapoints_loss, datapoints_accs, epoch, keepqs=[0,1])

# Plot (Accuracy vs Epochs includine lines for each Q) for each dimension
unique_dimensions = set([d[0][1] for d in data_losses])
for dimension in unique_dimensions:
    datapoints_loss = [d for d in data_losses if d[0][1] == dimension]
    datapoints_accs = [d for d in data_accs if d[0][1] == dimension]
    plot_accuracy_vs_epochs(datapoints_loss, datapoints_accs, dimension, keepqs=[0,1])
