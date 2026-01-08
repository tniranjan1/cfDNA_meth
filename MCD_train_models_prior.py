# libraries for data import and processing
import pandas as pd
import numpy as np
from MCD_data_generator import group_size
import MCD_data_import as mdi
import warnings
import pickle
import os

# disable pandas chained assignment warning (unnecessary for current df assignments)
pd.options.mode.chained_assignment = None
# ignore PerformanceWarning for short dataframes
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

##-----------------------------------------------------------------------------------------------##

# get methylation data
work_dir = '/results/ep/study/hg38s/study250-cfDNA_prelim/cfDNA-MS'

# check if beta_norm.pkl exists
if os.path.exists(work_dir + "/beta_norm.pkl"):
    with open(work_dir + "/beta_norm.pkl", "rb") as f:
        beta_norm = pickle.load(f)
else:
    MCD_beta = work_dir + "/MCD_reference/beta_values.txt"
    MS_beta = work_dir + "/MS_reference/MS_beta_values.txt"
    other_beta = work_dir + "/other_reference/other_beta_values.txt"
    blood_beta = work_dir + "/other_reference/blood_beta_values.txt"
    beta_norm = mdi.get_methylation_data([MCD_beta, MS_beta, other_beta, blood_beta])
    # store beta_norm for later use
    with open(work_dir + "/beta_norm.pkl", "wb") as f:
        pickle.dump(beta_norm, f)

##-----------------------------------------------------------------------------------------------##

# get phenotype labels

# check if combined_pheno_labels.pkl
if os.path.exists(work_dir + "/combined_pheno_labels.pkl"):
    with open(work_dir + "/combined_pheno_labels.pkl", "rb") as f:
        combined_pheno_labels = pickle.load(f)
else:
    MCD_pheno_file = work_dir + "/MCD_reference/pheno_label.txt"
    MS_pheno_file = work_dir + "/MS_reference/MS_pheno_label.txt"
    other_pheno_file = work_dir + "/other_reference/other_pheno_label.txt"
    blood_pheno_file = work_dir + "/other_reference/blood_pheno_label.txt"
    pheno_tables = {'MCD_pheno': mdi.load_phenotype_table(MCD_pheno_file),
                    'MS_pheno': mdi.load_phenotype_table(MS_pheno_file),
                    'other_pheno': mdi.load_phenotype_table(other_pheno_file),
                    'blood_pheno': mdi.load_phenotype_table(blood_pheno_file)}
    combined_pheno_labels = mdi.customize_and_merge_phenotype_labels(**pheno_tables)
    # store combined_pheno_labels for later use
    with open(work_dir + "/combined_pheno_labels.pkl", "wb") as f:
        pickle.dump(combined_pheno_labels, f)

##-----------------------------------------------------------------------------------------------##

# get top feature vector
if os.path.exists(work_dir + "/keep.pkl"):
    with open(work_dir + "/keep.pkl", "rb") as f:
        keep = pickle.load(f)
else:
    keep = mdi.find_top_features(beta_norm, combined_pheno_labels)
    # store keep for later use
    with open(work_dir + "/keep.pkl", "wb") as f:
        pickle.dump(keep, f)

##-----------------------------------------------------------------------------------------------##

import random

# randomize sample order (affects which samples are chosen for training vs validation)
random.seed(42)
samples = combined_pheno_labels.index.tolist()
random.shuffle(samples)
beta_norm = beta_norm.loc[samples,:]
combined_pheno_labels = combined_pheno_labels.loc[samples,:]

##-----------------------------------------------------------------------------------------------##

# libraries for model building and training
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from CLR.clr_callback import CyclicLR

# limit cores
tf_cores = 48

# set tensorflow threading
tf.function(jit_compile=True)
tf.config.optimizer.set_jit(True)

##-----------------------------------------------------------------------------------------------##

# other libraries
from p_tqdm import p_map
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

##-----------------------------------------------------------------------------------------------##

import MCD_data_generator as mdg
from MCD_optimize_models import study_training

# labels to use
these_labels = mdg.construct_label_combos()
train_size = mdg.get_training_sizes()
# max valid number of samples per label
max_allowed, max_valid = mdg.get_allowed_sizes(these_labels, train_size, combined_pheno_labels,max_size=60)

label_df = combined_pheno_labels
studies = {}
for label in these_labels:
    data_dict = mdg.data_generator(label, beta_norm, combined_pheno_labels,
                                  keep, max_allowed, max_valid, these_labels)
    studies[label] = study_training(**data_dict)

##-----------------------------------------------------------------------------------------------##

model_save_folder = "all_models_01_2026"
for m in range(len(models)):
    model_save_path = '_'.join(these_labels[m])
    model_save_path = model_save_folder + "/" + model_save_path
    models[m][0].save(model_save_path)  

model_save_path = model_save_folder + "/model_histories.pkl"
with open(model_save_path, 'wb') as out:
    pickle.dump([ [ m[1].epoch, m[1].history ] for m in models ], out, protocol=pickle.HIGHEST_PROTOCOL)

# for each model, history lin models, reconstruct the model with the lambda layer removed
new_models = []
for i in range(len(models)):
    old_model = models[i][0]
    label = these_labels[i]
    if len(label) == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    input_size = (keep > 0).sum()
    label_size = len(label)
    new_model = Sequential()
    new_model.add(layers.InputLayer(input_shape=(input_size,)))
    for layer in old_model.layers[1:-1]:
        new_model.add(layer)
    new_model.add(layers.Dense(label_size, activation=activation, name="output"))
    new_model.set_weights(old_model.get_weights())
    new_models.append([ new_model, models[i][1] ])

models = new_models

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('losses_accuracy.pdf') as pdf:
    for i in range(len(these_labels)):
        l = these_labels[i]
        history = models[i][1]
        figure_name = ", ".join(l)
        plt.figure(figsize=(18, 8))
        plt.subplot(1, 2, 1)
        epoch_range = history.epoch
        recall = history.history[[ h for h in history.history.keys() if ( 'recall' in h ) and ('val_recall' not in h) ][0]]
        precision = history.history[[ h for h in history.history.keys() if ( 'precision' in h ) and ('val_precision' not in h) ][0]]
        val_recall = history.history[[ h for h in history.history.keys() if ('val_recall' in h) ][0]]
        val_precision = history.history[[ h for h in history.history.keys() if ('val_precision' in h) ][0]]
        loss = history.history[[ h for h in history.history.keys() if ( 'loss' in h ) and ('val_loss' not in h) ][0]]
        val_loss = history.history[[ h for h in history.history.keys() if ('val_loss' in h) ][0]]
        plt.title('Training and Validation Recall/Precision')
        plt.plot(epoch_range, recall, label='Training Recall')
        plt.plot(epoch_range, precision, label='Training Precision')
        plt.plot(epoch_range, val_recall, label='Validation Recall')
        plt.plot(epoch_range, val_precision, label='Validation Precision')
        plt.legend(loc='lower right')
        plt.suptitle(figure_name, fontsize=16)
        plt.subplot(1, 2, 2)
        plt.title('Training and Validation Loss')
        plt.plot(epoch_range, loss, label='Training Loss')
        plt.plot(epoch_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        pdf.savefig()
        plt.close()

random.shuffle(samples)
beta_norm = beta_norm.loc[samples,:]
combined_pheno_labels = combined_pheno_labels.loc[samples,:]

predictions = []
for i in tqdm(range(len(these_labels))):
    model = models[i][0]
    prediction = model.predict(beta_norm.loc[:,keep > 0])
    predictions.append(prediction)

prediction = np.hstack(predictions)
prediction = pd.DataFrame(prediction, index=samples)
prediction_max = prediction.max(axis=0)
prediction_min = prediction.min(axis=0)
prediction = (prediction - prediction_min) / ( prediction_max - prediction_min )

beta_norm_normalizer = {}
beta_norm_normalizer['min'] = beta_min
beta_norm_normalizer['max'] = beta_max

prediction_normalizer = {}
prediction_normalizer['min'] = prediction_min
prediction_normalizer['max'] = prediction_max

label = these_labels[0]
loss = 'categorical_crossentropy'
activation = 'softmax'
regular = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
input_size = prediction.shape[1]
label_size = len(label)
combiner = Sequential([
    layers.Dropout(0.4, input_shape=(input_size,)),
    layers.Dense(label_size * 2, kernel_regularizer=regular, bias_regularizer=regular),
    layers.Dense(label_size, activation=activation, kernel_regularizer=regular, bias_regularizer=regular)
])
BATCH_SIZE = 32
epochs = 1000
main_label = combined_pheno_labels[label]
train_index = [ item for s in [ group_size(main_label.index[main_label[t] > 0], max_allowed) for t in label ] for item in s ]
train_index = pd.Index(train_index)
valid_index = [ b for b in df.index if b not in train_index ]
valid_index = pd.Index(valid_index)
valid_index = group_size(valid_index, max_valid)
valid_x = np.array(prediction.loc[valid_index])
valid_y = np.array(main_label.loc[valid_index])
sample_weight = main_label.loc[train_index]
class_weight = np.unique(np.array(sample_weight), return_counts=True, axis=0)
class_weight = [ class_weight[0], ( 1 / ( class_weight[1] / class_weight[1].sum() ) ) ]
sample_weight = np.array([ class_weight[1][abs(class_weight[0] - np.array(sample_weight)[s,:]).sum(axis=1) == 0][0] for s in range(len(sample_weight)) ])
steps_per_epoch = len(train_index) // BATCH_SIZE
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=5, min_lr=1e-7)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
clr = CyclicLR(base_lr=1e-9, max_lr=5e-3, step_size=2*steps_per_epoch, mode='triangular')
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
combiner.compile(loss = loss, optimizer = optimizer, metrics = metrics)
history = combiner.fit(
    x = np.array(prediction.loc[train_index]),
    y = np.array(main_label.loc[train_index]),
    epochs=epochs,
    validation_data=(valid_x, valid_y),
    batch_size=BATCH_SIZE,
    verbose=True,
    use_multiprocessing=True,
    workers=tf_cores,
    max_queue_size=tf_cores,
    callbacks = [ reduce_lr, clr, earlyStop ],
    sample_weight=sample_weight
)

label = these_labels[0]
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
activation = "sigmoid"
regular = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
input_size = prediction.shape[1]
label_size = len(label)
epochs = 1000
main_label = combined_pheno_labels[label]
train_index = [ item for s in [ group_size(main_label.index[main_label[t] > 0], max_allowed) for t in label ] for item in s ]
train_index = pd.Index(train_index)
valid_index = [ b for b in df.index if b not in train_index ]
valid_index = pd.Index(valid_index)
valid_index = group_size(valid_index, max_valid)
valid_x = np.array(prediction.loc[valid_index])
valid_y = np.array(main_label.loc[valid_index])
sample_weight = main_label.loc[train_index]
class_weight = np.unique(np.array(sample_weight), return_counts=True, axis=0)
class_weight = [ class_weight[0], ( 1 / ( class_weight[1] / class_weight[1].sum() ) ) ]
sample_weight = np.array([ class_weight[1][abs(class_weight[0] - np.array(sample_weight)[s,:]).sum(axis=1) == 0][0] for s in range(len(sample_weight)) ])
steps_per_epoch = len(train_index) // BATCH_SIZE
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=5, min_lr=1e-7)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
clr = CyclicLR(base_lr=1e-9, max_lr=1e-4, step_size=2*steps_per_epoch, mode='triangular')
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
combiner2 = Sequential([
    layers.Dropout(0.2, input_shape=(input_size,)),
    layers.Dense(label_size * 2, kernel_regularizer=regular, bias_regularizer=regular),
    layers.Dense(label_size, activation=activation, kernel_regularizer=regular, bias_regularizer=regular)
])
combiner2.compile(loss = loss, optimizer = optimizer, metrics = metrics)
history = combiner2.fit(
    x = np.array(prediction.loc[train_index]),
    y = np.array(main_label.loc[train_index]),
    epochs=epochs,
    validation_data=(valid_x, valid_y),
    batch_size=BATCH_SIZE,
    verbose=True,
    use_multiprocessing=True,
    workers=tf_cores,
    max_queue_size=tf_cores,
    callbacks = [ reduce_lr, clr, earlyStop ],
    sample_weight=sample_weight
)

model_save_folder = "all_models"
model_save_path = "combiner"
model_save_path = model_save_folder + "/" + model_save_path
combiner.save(model_save_path)

model_save_path = model_save_folder + "/combiner_history.pkl"
with open(model_save_path, 'wb') as out:
    pickle.dump([ history.epoch, history.history ], out, protocol=pickle.HIGHEST_PROTOCOL)

final_prediction = combiner.predict(prediction)
final_prediction = pd.DataFrame(final_prediction, columns=label, index=samples)
final_prediction_min = final_prediction.min(axis=1)
final_prediction_max = final_prediction.max(axis=1)
norm_prediction = final_prediction.subtract(final_prediction_min, axis=0).divide(final_prediction_max - final_prediction_min, axis=0)
final_normalizer = {}
final_normalizer['min'] = final_prediction_min
final_normalizer['max'] = final_prediction_max

normalizers = {}
normalizers['beta_norm'] = beta_norm_normalizer
normalizers['prediction_norm'] = prediction_normalizer

norm_save_path = model_save_folder + "/normalizers.pkl"
with open(norm_save_path, 'wb') as out:
    pickle.dump(normalizers, out, protocol=pickle.HIGHEST_PROTOCOL)

main_label = combined_pheno_labels[these_labels[0]]

def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

from sklearn.manifold import TSNE
# t-distributed stochastic neighbor embedding
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', verbose=2).fit_transform(norm_prediction)
colors = [ np.array(these_labels[0])[combined_pheno_labels[these_labels[0]].loc[s] > 0][0] for s in combined_pheno_labels.index ]
styles = [ 'o', '-', '|', 'x', 'v', '.', '+', 's', 'o', '-', '|', 'x', '^', ',', '+']
tSNE_df = pd.DataFrame(dict(comp1=rand_jitter(X_embedded[:,0]), comp2=rand_jitter(X_embedded[:,1]), group=colors))
tSNE_df = tSNE_df.sort_values('group')
styles = [ i for j in [ [ '-', '--', '-.', ':' ] for x in range(len(tSNE_df)) ] for i in j][:len(tSNE_df)]
xlim = [ tSNE_df.iloc[:,0].min(), tSNE_df.iloc[:,0].max() ]
xlim[0] = xlim[0] * 2

x_mean = [ tSNE_df[tSNE_df['group'] == m].iloc[:,0].median(axis=0) for m in main_label.columns ]
y_mean = [ tSNE_df[tSNE_df['group'] == m].iloc[:,1].median(axis=0) for m in main_label.columns ]
label = [ m for m in main_label.columns ]
labels = pd.DataFrame(dict(x=np.array(x_mean), y=np.array(y_mean), label=label))

def annotate_plot(frame, x_col, y_col, label_col, **kwargs):
    for label, x, y in zip(frame[label_col], frame[x_col], frame[y_col]):
        plt.annotate(label, xy=(x, y), **kwargs)

tSNE_df.set_index(['group', 'comp1']).unstack('group')['comp2'].plot(colormap='nipy_spectral', style='.', legend=False, figsize=(9, 9), ms=3)
#annotate_plot(labels, 'x', 'y', 'label')
plt.ylabel('comp2')
plt.savefig('tSNE.pdf')
plt.close()

tSNE_df = pd.DataFrame(dict(comp1=X_embedded[:,0], comp2=X_embedded[:,1], group=colors))
tSNE_df = tSNE_df.sort_values('group')
xlim = [ tSNE_df.iloc[:,0].min(), tSNE_df.iloc[:,0].max() ]
xlim[0] = xlim[0] * 2
x_mean = [ tSNE_df[tSNE_df['group'] == m].iloc[:,0].median(axis=0) for m in main_label.columns ]
y_mean = [ tSNE_df[tSNE_df['group'] == m].iloc[:,1].median(axis=0) for m in main_label.columns ]
label = [ m for m in main_label.columns ]
labels = pd.DataFrame(dict(x=x_mean, y=y_mean, label=label))
tSNE_df.set_index(['group', 'comp1']).unstack('group')['comp2'].plot(colormap='nipy_spectral', style='.', legend=False, figsize=(18, 18), ms=3)
annotate_plot(labels, 'x', 'y', 'label')
plt.ylabel('comp2')
plt.savefig('tSNE_noJitter.pdf')
plt.close()

# get weights
weights = []
for i in tqdm(range(len(these_labels))):
    tmp = models[i][0].layers[1].get_weights()[0]
    tmp = np.add(tmp, models[i][0].layers[1].get_weights()[1] / tmp.shape[0])
    tmp = np.matmul(tmp, models[i][0].layers[4].get_weights()[0])
    tmp = np.add(tmp, (models[i][0].layers[4].get_weights()[1] / np.array(tmp.shape).prod() ))
    weights.append(tmp)

dist_weights = np.hstack(weights)
dist_weights = dist_weights.reshape(np.array(dist_weights.shape).prod())
dist_weights.sort()
pdf = dist_weights / sum(abs(dist_weights))
pdf_pos = pdf[pdf >= 0]
pdf_neg = pdf[pdf < 0]
pdf_pos.sort()
pdf_neg = abs(pdf_neg)
pdf_neg.sort()
pdf_neg = pdf_neg * -1
pdf_pos = np.flip(pdf_pos)
pdf_neg = np.flip(pdf_neg)
cdf = abs(pdf)
cdf.sort()
cdf = np.cumsum(np.flip(cdf))
plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
plt.plot(range(len(pdf_pos)), pdf_pos, 'r-', label="PDF")
plt.plot(range(len(pdf_neg)), pdf_neg, 'b-', label="PDF")
plt.title('PDF')
plt.subplot(1, 2, 2)
plt.plot(range(len(cdf)), cdf, 'b-', label="CDF")
plt.title('CDF')
plt.savefig('pdf_cdf.pdf')
plt.close()

import pysam
import pybedtools
sites = pd.DataFrame([ s.split('_') for s in beta_norm.columns[keep > 0] ])
sites.columns = [ 'CHR', 'START', 'STOP' ]
sites['START'][sites['START'] == 'NA'] = '0'
sites['START'] = sites['START'].astype(int) + 1
sites['STOP'][sites['STOP'] == 'NA'] = '1'
sites['STOP'] = sites['STOP'].astype(int)
sites.index = beta_norm.columns[keep > 0]
sites_as_bed = pybedtools.BedTool.from_dataframe(sites)
sites_order = pd.Series(range(len(sites)), index=sites.index)

bam_files = "/home/tejasvi/cfDNA-MS/cfDNA_samples"
bam_files = [ bam_files + "/" + o for o in os.listdir(bam_files) if o.endswith('.bam') ]
cfDNA_path = '/home/tejasvi/cfDNA-MS/methylation-calls/methylation_frequency'
cfDNA_methylation_files_path = [ cfDNA_path + "/" + o for o in os.listdir(cfDNA_path) if o.endswith('.tsv') ]
total = []
coverage = []
beds = []
for b in tqdm(cfDNA_methylation_files_path):
    file_as_tab = pd.read_table(b)
    change_col= file_as_tab.columns.to_list()
    change_col[:3] = [ 'CHR', 'START', 'STOP' ]
    file_as_tab.columns = change_col
    file_as_tab = file_as_tab.iloc[np.array([ '_' not in f for f in file_as_tab['CHR'] ]),:]
    total.append( len(file_as_tab) )
    file_as_tab['STOP'] = file_as_tab['STOP'] + 2
    file_as_tab.index = file_as_tab['CHR'] + '_' + file_as_tab['START'].astype(str) + '_' + file_as_tab['STOP'].astype(str)
    file_as_tab['START'] = file_as_tab['START'] + 1
    intersection = sites.index.intersection(file_as_tab.index)
    file_as_tab = file_as_tab.loc[sites_order.loc[intersection].sort_values().index]
    beds.append(file_as_tab)
    coverage.append(len(file_as_tab))

coverages = np.array(coverage)
totals = np.array(total) / ( 3.1e9 * .21 * .21 * .2 )
groups = [ ('_'.join(b.replace('meth-nanopolish-sminimap2-', '').replace('merged_', '').split('/')[-1].split('_')[:-2]).split('_v')[0]) for b in cfDNA_methylation_files_path ]
cov_fraction = coverages / (keep > 0).sum()
x = np.arange(len(groups))
width = 0.35
fig, ax = plt.subplots(figsize=(24, 8))
rects1 = ax.bar(x - width/2, totals, width, label='genomeFracCov')
rects2 = ax.bar(x + width/2, cov_fraction, width, label='informativeFracCov')
ax.set_ylabel('Fractional Cov')
ax.set_title('Fractional Coverage by CSF Sample')
ax.set_xticks(x, groups)
plt.xticks(rotation=45)
ax.legend()
ax.bar_label(rects1, padding=3, fmt="%.2f")
ax.bar_label(rects2, padding=3, fmt="%.2f")
fig.tight_layout()
plt.savefig('fractional_coverage.pdf')
plt.close()

# get weights
focused_weights = np.hstack(weights)
combined_weights = np.matmul(focused_weights, combiner.layers[1].get_weights()[0])
combined_weights = np.add(combined_weights, ( combiner.layers[1].get_weights()[1] / np.array(combined_weights.shape).prod() ))
combined_weights = np.matmul(combined_weights, combiner.layers[2].get_weights()[0])
combined_weights = np.add(combined_weights, ( combiner.layers[2].get_weights()[1] / np.array(combined_weights.shape).prod() ))

checker = combined_weights > 0
pmt = []
for i in range(checker.shape[1]):
    tmp = []
    for j in range(checker.shape[1]):
        tmp.append((checker[:,i] == checker[:,j]).sum())
    pmt.append(np.array(tmp))

pmt = np.vstack(pmt) / len(checker)

sum_results = []
for c in tqdm(range(combined_weights.shape[1])):
    sample_results = []
    for s in samples:
        use = beta_norm.loc[s,keep > 0]
        use = np.array(use)
        sample_results.append( (use * combined_weights[:,c]).sum() )
    sum_results.append(np.array(sample_results))

sum_results = np.vstack(sum_results).transpose()
sum_results = pd.DataFrame(sum_results, index=samples, columns=these_labels[0])
sample_order = [ i for l in [ combined_pheno_labels.index[combined_pheno_labels[t] > 0].tolist() for t in these_labels[0] ] for i in l ]
sum_results = sum_results.loc[sample_order]
sum_results = sum_results.subtract(sum_results.min(axis=1), axis=0)
sum_results = sum_results.divide(sum_results.max(axis=1), axis=0)

cmap = plt.cm.hot
norm = plt.Normalize(sum_results.min().min(), sum_results.max().max())
rgba = cmap(norm(sum_results))
padded_rgba = []
for t in these_labels[0]:
    to_shift = np.array([ s[0] for s in enumerate(sample_order) if s[1] in combined_pheno_labels.index[combined_pheno_labels[t] > 0] ])
    shifted_rgba = rgba[to_shift,:,:]
    to_append = np.zeros((1, len(these_labels[0]), 4))
    to_append[:,:,2] = 1
    to_append[:,:,3] = 1
    shifted_rgba = np.append(shifted_rgba, to_append, axis=0)
    padded_rgba.append(shifted_rgba)

fig, ax = plt.subplots(figsize=(24, 12))
ax.imshow(np.vstack(padded_rgba), interpolation='nearest', aspect='auto')
labels = these_labels[0]
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_title("Weighted scoring")
ax.set_xlabel('Sampling weighting source')
ax.set_ylabel('Sample groups')
ax.set_yticklabels(labels)

def ytick_val(t):
    u = np.array([ s[0] for s in enumerate(sample_order) if s[1] in combined_pheno_labels.index[combined_pheno_labels[t] > 0] ]).mean()
    v = [ l[0] for l in enumerate(labels) if l[1] == t ][0]
    return u + v

yticks = [ ytick_val(t) for t in labels ]
ax.set_yticks(yticks)
plt.savefig('heatmap.pdf')
plt.close()

## analyse cfDNA sample methylation
from scipy.special import softmax
from scipy.special import expit as sigmoid

weights_per_sample = []
for r in tqdm(range(len(groups))):
    g = groups[r]
    methyl = beds[r]
    site_locations = np.array(sites_order[methyl.index])
    beta = np.array(methyl['methylated_frequency'])
    list_of_weights = []
    for m in range(len(models)):
        weighting = models[m][0].layers[1].get_weights()[0]
        mean_weighting = weighting.mean(axis=0)
        weighting = weighting[site_locations]
        mean_current = weighting.mean(axis=0)
        weighting = weighting.transpose()
        weight = beta * weighting
        weight = weight.sum(axis=1) * ( (keep > 0).sum() / len(site_locations) )
        weight = weight / (mean_current / mean_weighting)
        weighting = models[m][0].layers[1].get_weights()[1]
        weight = weight + weighting
        weighting = models[m][0].layers[4].get_weights()[0]
        weighting = weighting.transpose()
        weight = weight * weighting
        weight = weight.sum(axis=1)
        weighting = models[m][0].layers[4].get_weights()[1]
        weight = weight + weighting
        if len(weight) > 1:
            weight = softmax(weight)
        else:
            weight = sigmoid(weight)
        list_of_weights.append(weight)
    weight = np.concatenate(list_of_weights)
    weighting = combiner.layers[1].get_weights()[0]
    weighting = weighting.transpose()
    weight = weight * weighting
    weight = weight.sum(axis=1)
    weighting = combiner.layers[1].get_weights()[1]
    weight = weight + weighting
    weighting = combiner.layers[2].get_weights()[0]
    weighting = weighting.transpose()
    weight = weight * weighting
    weight = weight.sum(axis=1)
    weighting = combiner.layers[2].get_weights()[1]
    weight = weight + weighting
    weight = softmax(weight)
    weights_per_sample.append(weight)

color_groups = np.array(groups)
non_MS = [ 1, 2, 3, 4, 6, 7 ]
for n in non_MS: color_groups[color_groups == ('MS_CSF_' + str(n))] = 'CSF_nonMS'

color_groups[np.array([ 'MS_CSF' in c for c in color_groups ])] = 'CSF_MS'
color_groups[np.array([ '_CO' in c for c in color_groups ])] = 'Control'
color_groups[np.array([ not ( ('Control' in c) or ('CSF_' in c) ) for c in color_groups ])] = 'CVD19'
main_groups = color_groups
main_groups[main_groups == 'CVD19'] = 'serum_nonMS'
main_groups[main_groups == 'Control'] = 'serum_nonMS'

weights = pd.DataFrame(np.vstack(weights_per_sample))
weights.index = groups
weights.columns = these_labels[0]
weights['group'] = main_groups

low_coverage_outliers = [ 'MS_CSF_7', 'tvdb_CO_0306', 'tvdb_CO_0308' ]

norm_weights = weights.copy().transpose()
for o in low_coverage_outliers: norm_weights.pop(o)

norm_weights = norm_weights.transpose()
for t in these_labels[0]:
    norm_weights[t] = norm_weights[t] - norm_weights[t].mean()
    norm_weights[t] = norm_weights[t] / norm_weights[t].std()

X = np.arange(len(these_labels[0]))
for_jit = [ i for l in [ [0,.1] for h in range(len(X)) ] for i in l ]
data_plot_x = []
data_plot_y = []
data_plot_c = []
for i in range(len(these_labels[0])):
    t = these_labels[0][i]
    x = X[i] - 0.25
    y = norm_weights[t].loc[norm_weights.index[norm_weights['group'] == 'serum_nonMS']].tolist()
    c = 'r'
    data_plot_x.append(x)
    data_plot_y.append(y)
    data_plot_c.append(c)
    x = X[i] + 0
    y = norm_weights[t].loc[norm_weights.index[norm_weights['group'] == 'CSF_MS']].tolist()
    c = 'b'
    data_plot_x.append(x)
    data_plot_y.append(y)
    data_plot_c.append(c)
    x = X[i] + 0.25
    y = norm_weights[t].loc[norm_weights.index[norm_weights['group'] == 'CSF_nonMS']].tolist()
    c = 'g'
    data_plot_x.append(x)
    data_plot_y.append(y)
    data_plot_c.append(c)

## check significance
from scipy import stats
all_t = []
all_p = []
for i in (np.array(range(len(these_labels[0]))) * 3):
    j = i + 1
    t, p = stats.ttest_ind(data_plot_y[i], data_plot_y[j] + data_plot_y[j+1])
    all_t.append(t)
    all_p.append(p)

colors = [ x for y in [ [ 'firebrick', 'mediumslateblue', 'forestgreen' ] for x in range(len(these_labels[0])) ] for x in y ]
fig, ax = plt.subplots(figsize=(18, 12))
first_y = [ data_plot_y[r*3] for r in range(len(these_labels[0])) ]
first_x = [ data_plot_x[r*3] for r in range(len(these_labels[0])) ]
bplot1 = ax.boxplot(first_y, positions=first_x, widths = 0.15, patch_artist = True, notch=True, boxprops=dict(facecolor="firebrick"))
second_y = [ data_plot_y[(r*3)+1] for r in range(len(these_labels[0])) ]
second_x = [ data_plot_x[(r*3)+1] for r in range(len(these_labels[0])) ]
bplot2 = ax.boxplot(second_y, positions=second_x, widths = 0.15, patch_artist = True, notch=True, boxprops=dict(facecolor="mediumslateblue"))
third_y = [ data_plot_y[(r*3)+2] for r in range(len(these_labels[0])) ]
third_x = [ data_plot_x[(r*3)+2] for r in range(len(these_labels[0])) ]
bplot3 = ax.boxplot(third_y, positions=third_x, widths = 0.15, patch_artist = True, notch=True, boxprops=dict(facecolor="forestgreen"))
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

ax.legend([bplot1["boxes"][0], bplot2["boxes"][0], bplot3["boxes"][0]], ['Covid19-serum', 'MS-CSF', 'nonMS-CSF'], loc='upper right')
ax.set_xticks(X, these_labels[0])
plt.xticks(rotation=45)
plt.savefig('relative_deconv_2.pdf')
plt.close()

## using just weights
weighted_results = []
for c in tqdm(range(combined_weights.shape[1])):
    sample_results = []
    for methyl in beds:
        site_locations = np.array(sites_order[methyl.index])
        beta = np.array(methyl['methylated_frequency'])
        use = combined_weights[site_locations,c]
        value = (use * beta).sum()
        value = value * ( len(sites_order) / len(use) )
        sample_results.append(value)
    weighted_results.append(np.array(sample_results))

weighted_results = np.vstack(weighted_results).transpose()
weighted_results = pd.DataFrame(weighted_results, index=groups, columns=these_labels[0])
weighted_results['group'] = main_groups
weighted_results = weighted_results.sort_values('group')
weighted_results = weighted_results[these_labels[0]]
weighted_results = weighted_results.subtract(weighted_results.min(axis=1), axis=0)
weighted_results = weighted_results.divide(weighted_results.max(axis=1), axis=0)

cmap = plt.cm.hot
norm = plt.Normalize(weighted_results.min().min(), weighted_results.max().max())
rgba = cmap(norm(weighted_results))

fig, ax = plt.subplots(figsize=(24, 12))
ax.imshow(rgba, interpolation='nearest', aspect='auto')
labels = these_labels[0]
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_title("Weighted scoring")
ax.set_xlabel('Sampling weighting source')
ax.set_ylabel('Sample groups')
plt.savefig('heatmap_weird.pdf')
plt.close()