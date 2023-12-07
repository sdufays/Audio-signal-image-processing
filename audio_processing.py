"""
6.101 Lab 0:
Audio Processing
"""

import wave
import struct

# No additional imports allowed!


def backwards(sound):
    """
    Returns a new sound containing the samples of the original in reverse
    order, without modifying the input sound.

    Args:
        sound: a dictionary representing the original mono sound

    Returns:
        A new mono sound dictionary with the samples in reversed order
    """
    rate = sound["rate"]  # extract rate
    samples = sound["samples"]  # extract samples
    reversed_sound = samples[::-1]

    return {"rate": rate, "samples": reversed_sound}


def mix(sound1, sound2, p):
    """
    Mixes two sounds (inputs) using mixing parameter p
    If the two sounds have different sampling rates, returns None
    The resulting sound needs to have a length equal to the shortest input sound

    Args:
        sound1: a dictionary representing the first original mono sound
        sound2: a dictionary representing the second original mono sound

    Returns:
        A new mono sound dictionary that mixed both sounds according to the
        mixing parameter, p
    """
    if not ("rate" in sound1 and "rate" in sound2 and sound1["rate"] == sound2["rate"]):
        return None

    rate = sound1["rate"]

    min_length = min(len(sound1["samples"]), len(sound2["samples"]))

    mixed_samples = []
    for i in range(min_length):
        s1 = p * sound1["samples"][i]
        s2 = (1 - p) * sound2["samples"][i]
        mixed_samples.append(s1 + s2)

    return {"rate": rate, "samples": mixed_samples}


def echo(sound, num_echoes, delay, scale):
    """
    Compute a new signal consisting of several scaled-down and delayed versions
    of the input sound. Does not modify input sound.

    Args:
        sound: a dictionary representing the original mono sound
        num_echoes: int, the number of additional copies of the sound to add
        delay: float, the amount of seconds each echo should be delayed
        scale: float, the amount by which each echo's samples should be scaled

    Returns:
        A new mono sound dictionary resulting from applying the echo effect.
    """
    sample_delay = round(delay * sound["rate"])
    new_samples_length = len(sound["samples"]) + num_echoes * sample_delay
    new_samples = [0] * new_samples_length

    for i, sample in enumerate(sound["samples"]):
        new_samples[i] += sample

    for echo_num in range(1, num_echoes + 1):
        current_scale = scale**echo_num
        for i, sample in enumerate(sound["samples"]):
            new_samples[i + echo_num * sample_delay] += sample * current_scale

    return {"rate": sound["rate"], "samples": new_samples}


def pan(sound):
    """
    Adjusts the volume in the left and right channels separately to create a
    spatial effect

    Args:
        sound: A dictionary representing a stereo sound

    Returns:
        A new stereo sound dictionary with the panning effect
    """
    assert len(sound["left"]) == len(sound["right"])
    num = len(sound["left"])
    new_left = []
    new_right = []

    for i in range(num):
        scale_factor = i / (num - 1)
        new_left_sample = sound["left"][i] * (1 - scale_factor)
        new_right_sample = sound["right"][i] * scale_factor
        new_left.append(new_left_sample)
        new_right.append(new_right_sample)

    return {"rate": sound["rate"], "left": new_left, "right": new_right}


def remove_vocals(sound):
    """
    Removes  vocals from a stereo sound to produce a mono sound

    Args:
        sound: A dictionary representing a stereo sound

    Returns:
        A new mono sound dictionary with the vocals removed

    """
    new_samples = []

    for left_sample, right_sample in zip(sound["left"], sound["right"]):
        new_sample = left_sample - right_sample
        new_samples.append(new_sample)

    return {"rate": sound["rate"], "samples": new_samples}


# below are helper functions for converting back-and-forth between WAV files
# and our internal dictionary representation for sounds


def load_wav(filename, stereo=False):
    """
    Load a file and return a sound dictionary.

    Args:
        filename: string ending in '.wav' representing the sound file
        stereo: bool, by default sound is loaded as mono, if True sound will
            have left and right stereo channels.

    Returns:
        A dictionary representing that sound.
    """
    sound_file = wave.open(filename, "r")
    chan, bd, sr, count, _, _ = sound_file.getparams()

    assert bd == 2, "only 16-bit WAV files are supported"

    out = {"rate": sr}

    left = []
    right = []
    for i in range(count):
        frame = sound_file.readframes(1)
        if chan == 2:
            left.append(struct.unpack("<h", frame[:2])[0])
            right.append(struct.unpack("<h", frame[2:])[0])
        else:
            datum = struct.unpack("<h", frame)[0]
            left.append(datum)
            right.append(datum)

    if stereo:
        out["left"] = [i / (2**15) for i in left]
        out["right"] = [i / (2**15) for i in right]
    else:
        samples = [(ls + rs) / 2 for ls, rs in zip(left, right)]
        out["samples"] = [i / (2**15) for i in samples]

    return out


def write_wav(sound, filename):
    """
    Save sound to filename location in a WAV format.

    Args:
        sound: a mono or stereo sound dictionary
        filename: a string ending in .WAV representing the file location to
            save the sound in
    """
    outfile = wave.open(filename, "w")

    if "samples" in sound:
        # mono file
        outfile.setparams((1, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = [int(max(-1, min(1, v)) * (2**15 - 1)) for v in sound["samples"]]
    else:
        # stereo
        outfile.setparams((2, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = []
        for l_val, r_val in zip(sound["left"], sound["right"]):
            l_val = int(max(-1, min(1, l_val)) * (2**15 - 1))
            r_val = int(max(-1, min(1, r_val)) * (2**15 - 1))
            out.append(l_val)
            out.append(r_val)

    outfile.writeframes(b"".join(struct.pack("<h", frame) for frame in out))
    outfile.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place to put your
    # code for generating and saving sounds, or any other code you write for
    # testing, etc.

    # here is an example of loading a file (note that this is specified as
    # sounds/hello.wav, rather than just as hello.wav, to account for the
    # sound files being in a different directory than this file)

    ## example code
    # hello = load_wav("sounds/hello.wav")
    # write_wav(backwards(hello), "hello_reversed.wav")

    ## testing code for reversing audio
    # mystery = load_wav("audio_processing/sounds/mystery.wav")
    # write_wav(backwards(mystery), "mystery_reversed.wav")

    ## testing code for mixing audio
    # synth = load_wav("sounds/synth.wav")
    # water = load_wav("sounds/water.wav")
    # write_wav(mix(synth, water, 0.2), "mixed_sound.wav")

    # # testing code for the echo
    # chord = load_wav("sounds/chord.wav")
    # write_wav(echo(chord, 5, 0.3, 0.6), "echo_chord.wav")

    ## testing code for pan
    # car = load_wav("sounds/car.wav", stereo=True)
    # write_wav(pan(car), "pan_car.wav")

    ## testing code for removing vocals
    # lookout = load_wav("sounds/lookout_mountain.wav", stereo=True)
    # write_wav(remove_vocals(lookout), "lookout_remove_vocals.wav")

    print("Testing starts here: ")
