# model checkpoint and store the tensor
import datetime
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau


def complete_words(st, true):
    st, true = str(st), str(true)
    st_split = st.split()
    for word_2 in range(len(st_split)):
        for word in true.lower().split():
            if st_split[word_2] in word:
                st_split[word_2] = word

    st = ' '.join(st_split)
    return st.lower()


# https://stackoverflow.com/questions/51889378/how-to-use-keras-reducelronplateau
def model_check_point_tensor_board(fold):
    # val_loss is the sum of val_activation_loss and val_activation_1_loss.
    # Those two losses are the losses associated with predicting selected_text start token and selected_text end token.
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        '%s-roberta-%i.h5' % ("v0", fold), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')

    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_grads=True)

    return [checkpointer, tensorboard_callback]