import pandas as pd
import numpy as np
from IPython.display import Audio
import pywt
from scipy import signal
from scipy.io import wavfile
from scipy.signal import resample
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.fft import fftshift
from sklearn.metrics import roc_curve, roc_auc_score
from skimage.restoration import denoise_wavelet
import wfdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

def process_signal(senal, fs):
    # filtro de 50 Hz
    fc1 = 50.0
    Q1 = fc1 / 0.2  # "fc/bw" donde el ancho de banda en filtros biológicos es 0.2 o 0.1 a cada lado de fc
    b1, a1 = signal.iirnotch(fc1, Q=Q1, fs=fs)
    # filtro de 60 Hz
    fc2 = 60.0
    Q2 = fc2 / 0.2
    b2, a2 = signal.iirnotch(fc2, Q=Q2, fs=fs)
    # filtro pasabajas butter
    fc3 = [20, 400]
    orden = 4
    b3, a3 = signal.butter(orden, fc3, btype='band', analog=False, fs=fs, output='ba')
    # aplica filtros
    senal = signal.lfilter(b1, a1, senal)
    senal = signal.lfilter(b2, a2, senal)
    senal = signal.lfilter(b3, a3, senal)

    # aplica denoiser basado en wavelet filter banks
    senal = denoise_wavelet(senal, method='VisuShrink', mode='soft', wavelet_levels=10, wavelet='sym8',
                            rescale_sigma='True')
    denoised = senal

    return denoised


def has_nan(senal):
    nan_ = np.isnan(senal.sum())
    return nan_


# funciona
def leer_wav(filename):
    data = wavfile.read(filename)
    # Separeate the object elements
    rate = data[0]
    audio = data[1]
    nan = has_nan(audio)
    return audio, rate, nan


target_size = (128, 128)  # aumentar el tamaño para observar los detalles, potencia de dos puede ser 128


def procesar(dataset, id, freq_cut=100, target_size=target_size, time_cut=150):
    audio, rate, nan = leer_wav('./{}/{}.wav'.format(dataset, id))  # llamada de la funcion
    audio_dnsd = process_signal(audio, rate)
    nan = has_nan(audio_dnsd)
    f, t, Sxx = signal.spectrogram(audio_dnsd, rate)
    if Sxx.shape[0] > freq_cut and Sxx.shape[1] > time_cut:
        return resize(Sxx[:freq_cut, :time_cut], target_size, anti_aliasing=True), nan


murmuro = pd.read_csv('murmur_vector - murmur_vector.csv')

entrenamientos = ["training-a"]
nan = False
X = []
y = []
pids = []  # ids para recuperar la data original
k = 0
mumur = []
nomumur = []
listanan = []
dataset = "training-a"

for i in murmuro['Presencia de Murmuros']:
    pid = murmuro['Id'][k]
    if i == 1:
        mumur.append(pid)
    else:
        nomumur.append(pid)
    k += 1
print(mumur)
print(nomumur)

for pid in nomumur:
    try:
        x, nan = procesar(dataset, pid)  # llamada de la funcion
        if nan == False:
            X.append(x)
            y.append(0)
            pids.append(pid)
            # casos con soplo o murmuro
    except:
        pass

for pid in mumur:
    try:
        x, nan = procesar(dataset, pid)  # llamada de la funcion
        if (nan == False):
            X.append(x)
            y.append(1)
            pids.append(pid)
    except:
        pass

X = np.array(X)
X = np.log1p(X)
y = np.array(y)
pids = np.array(pids)

# Ajustamos las dimensiones de las imágenes
if X.ndim == 3:  # si la dimension es 3, añade una
    X = X[..., None]
print(X.shape)

X_train, X_test, y_train, y_test, pids_train, pids_test = train_test_split(X, y, pids, test_size=0.2, random_state=42)

from tensorflow.keras import datasets, layers, models, optimizers, Input, Model
from tensorflow import keras


signal = keras.Input(shape=(128,128,1))

x = layers.Conv2D(16, 3, padding = 'same')(signal)
x = layers.Conv2D(32, 3, padding = 'same')(x)
x = layers.Conv2D(32, 5, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64, 5, padding = 'same')(x)
x = layers.Conv2D(64, 5, padding = 'same')(x)
x = layers.Conv2D(64, 5, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(32, 5, padding = 'same')(x)
x = layers.Conv2D(32, 3, padding = 'same')(x)
x = layers.Conv2D(16, 3, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x) #layers.GlobalMaxPooling2D()(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(2, activation='sigmoid')(x)

model = keras.Model(signal, x)
model.summary()

lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate= 0.001, decay_steps= 30, decay_rate= 0.85) # 0.75

model.compile(optimizer= optimizers.Adam(learning_rate=lr_schedule, decay=1e-6),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

import os

path = 'pcg_model/training_2/'
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = path + "cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 15
epochs = 35
iterations = int(np.ceil(X_train.shape[0]/batch_size))
print("Iterations: ", iterations)

# Create a callback that saves the model's weights every n epochs
n = 1
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=n*iterations)

# Train the model with the new callback
hist = model.fit(X_train,
          y_train,
          batch_size = batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          callbacks=[cp_callback])  # Pass callback to training

model.save('pcg_model/pcgmodel2_cnn.h5')

import pickle


# save:
f = open(path + 'history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()

# retrieve:
import pickle
#path='pcg_model/training_3/'
f = open(path + 'history.pckl', 'rb')
history = pickle.load(f)
f.close()

#hist.history
best = history['val_accuracy'].index(max(history['val_accuracy']))
print("Best accuracy in cp-00"+str(best+1),"=",max(history['val_accuracy']))

from keras.models import load_model
model_new = load_model('pcg_model/pcgmodel2_cnn.h5')
#path + "cp-00"+str(best+1)+".ckpt"
checkpoint_selected = path + "cp-00"+str(best+1)+".ckpt"
model_new.load_weights(checkpoint_selected)

loss, acc = model_new.evaluate(X_test, y_test, verbose=1)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#loss
plt.figure()
plt.title("Loss x epoch")
plt.plot(history['loss'], label='train')
plt.plot(history['val_loss'], label='test')
plt.xlabel('epoch');
plt.ylabel('loss')
plt.legend()
plt.show()

#Accuracy
# Include Learning rate
import numpy as np

steps_per_epoch = 22
initial_learning_rate_var = 0.001 #0.0025
decay_steps_var = 30 # 525
decay_rate_var = 0.85 #0.75

epoch = np.arange(0,35,0.01)

lr_curve = np.multiply(initial_learning_rate_var, np.power(decay_rate_var,np.multiply(epoch,steps_per_epoch/decay_steps_var)))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

lns1 = ax1.plot(history['accuracy'], label='train', color='tab:orange')
lns2 = ax1.plot(history['val_accuracy'], label='test', color='tab:blue')
lns3 = ax2.plot(epoch, lr_curve,'--', color='tab:green', label='learning rate')

# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='center right')

ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax2.set_ylabel('learning rate')  # we already handled the x-label with ax1

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title("Accuracy x epoch")
plt.show()

#printea los errores
y_pred = model_new.predict(X_test).argmax(axis=1)
errors = np.where(y_pred != y_test)
print(errors)
print(pids[errors])


fpr, tpr, th = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.title("Curve ROC and AUC")
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc=4)
plt.show()

#definir las clases
cnn_labels = ['nomumur','murmur']

#Confusion Matrix
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true = y_test, y_pred = y_pred)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals = 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plot_confusion_matrix(cm=cm, classes=cnn_labels, normalize = False, title='Confusion Matrix')


#plots
c_matrix = cm

lst_acc = []  # Accuracy
lst_pc = []  # Precision
lst_rc = []  # Recall
lst_sp = []  # Specificity
lst_fs = []  # F-score

for i in range(len(cnn_labels)):
    TP = c_matrix[i, i];
    FN = np.sum(c_matrix[i, :]) - c_matrix[i, i];
    FP = np.sum(c_matrix[:, i]) - c_matrix[i, i];
    TN = np.sum(c_matrix) - TP - FP - FN;

    acc = (TP + TN) / (TP + TN + FN + FP)
    pc = TP / (TP + FP)
    rc = TP / (TP + FN)
    sp = TN / (TN + FP)
    fs = (2 * pc * rc) / (pc + rc)

    lst_acc.append(acc)
    lst_pc.append(pc)
    lst_rc.append(rc)
    lst_sp.append(sp)
    lst_fs.append(fs)

print("\t \t", cnn_labels)
print('Accuracy (%)\t', np.round(np.array(lst_acc) * 100, decimals=2))
print('Precision (%)\t', np.round(np.array(lst_pc) * 100, decimals=2))
print('Recall (%)\t', np.round(np.array(lst_rc) * 100, decimals=2))
print('Specificity (%)\t', np.round(np.array(lst_sp) * 100, decimals=2))
print('F-score (%)\t', np.round(np.array(lst_fs) * 100, decimals=2))

