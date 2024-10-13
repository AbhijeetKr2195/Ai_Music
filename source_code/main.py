import numpy as np
import glob
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

def get_notes():
    notes = []
    for file in glob.glob("midi_songs/*.mid"):
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    return notes

def prepare_sequences(notes, n_vocab):
    sequence_length = 100
    pitchnames = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    network_input = []
    network_output = []
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[note] for note in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
    network_output = to_categorical(network_output)
    return network_input, network_output

lstm_units_1 = 2486
lstm_units_2 = 2489
lstm_units_3 = 1024
lstm_units_4 = 1024
dense_units = 1024
dropout_rate = 0.2
optimizer = 'adam'
loss_function = 'categorical_crossentropy'
epochs = 50
batch_size = 128

def create_deep_lstm_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(lstm_units_1, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units_2, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(lstm_units_3, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units_4))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss=loss_function, optimizer=optimizer)
    return model

def train_model(model, network_input, network_output):
    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    ]
    model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    model.save('music_generation_model.keras')

def generate_notes(model, network_input, pitchnames, n_vocab, int_to_note, duration_seconds=60, bpm=80, beam_width=3, temperature=7.0):
    beats_per_second = bpm / 60
    notes_per_second = beats_per_second
    total_notes = int(duration_seconds * notes_per_second)
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []
    for _ in range(total_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        predictions = model.predict(prediction_input, verbose=0)[0]
        predictions = np.log(predictions + 1e-8) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        top_indices = np.argsort(predictions)[-beam_width:]
        probabilities = predictions[top_indices] / np.sum(predictions[top_indices])
        index = np.random.choice(top_indices, p=probabilities)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern, index)[1:]
    return prediction_output

def create_midi(prediction_output, instrument_name='BassDrum'):
    offset = 0
    output_notes = []
    try:
        chosen_instrument = instrument.fromString(instrument_name)
    except Exception as e:
        print(f"Invalid instrument name: {instrument_name}. Defaulting to 'Piano'.")
        chosen_instrument = instrument.Piano()
    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.storedInstrument = chosen_instrument
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = chosen_instrument
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')

n_vocab = len(set(notes))
network_input, network_output = prepare_sequences(notes, n_vocab)
model = create_deep_lstm_model(network_input, n_vocab)
pitchnames = sorted(set(notes))
int_to_note = {number: note for number, note in enumerate(pitchnames)}
prediction_output = generate_notes(model, network_input, pitchnames, n_vocab, int_to_note)
create_midi(prediction_output)
