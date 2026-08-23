/*
 * Windows-friendly FT8 WAV generator wrapper for kgoba/ft8_lib.
 * The modulation follows demo/gen_ft8.c from ft8_lib (MIT license), but uses
 * heap buffers instead of C variable-length arrays so it builds with MSVC.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "common/wave.h"
#include "ft8/constants.h"
#include "ft8/encode.h"
#include "ft8/message.h"

#define FT8_SYMBOL_BT 2.0f
#define GFSK_CONST_K 5.336446f

static void gfsk_pulse(int n_spsym, float symbol_bt, float* pulse)
{
    int i;
    for (i = 0; i < 3 * n_spsym; ++i) {
        float t = i / (float)n_spsym - 1.5f;
        float arg1 = GFSK_CONST_K * symbol_bt * (t + 0.5f);
        float arg2 = GFSK_CONST_K * symbol_bt * (t - 0.5f);
        pulse[i] = (erff(arg1) - erff(arg2)) / 2.0f;
    }
}

static int synth_gfsk(const uint8_t* symbols, int n_sym, float f0,
                      float symbol_period, int signal_rate, float* signal)
{
    int i, j, k;
    int n_spsym = (int)(0.5f + signal_rate * symbol_period);
    int n_wave = n_sym * n_spsym;
    int n_dphi = n_wave + 2 * n_spsym;
    float dphi_peak = 2.0f * (float)M_PI / n_spsym;
    float* dphi = (float*)malloc((size_t)n_dphi * sizeof(float));
    float* pulse = (float*)malloc((size_t)(3 * n_spsym) * sizeof(float));
    float phi = 0.0f;
    int n_ramp = n_spsym / 8;
    if (dphi == NULL || pulse == NULL) {
        free(dphi);
        free(pulse);
        return 0;
    }
    for (i = 0; i < n_dphi; ++i) {
        dphi[i] = 2.0f * (float)M_PI * f0 / signal_rate;
    }
    gfsk_pulse(n_spsym, FT8_SYMBOL_BT, pulse);
    for (i = 0; i < n_sym; ++i) {
        int ib = i * n_spsym;
        for (j = 0; j < 3 * n_spsym; ++j) {
            dphi[j + ib] += dphi_peak * symbols[i] * pulse[j];
        }
    }
    for (j = 0; j < 2 * n_spsym; ++j) {
        dphi[j] += dphi_peak * pulse[j + n_spsym] * symbols[0];
        dphi[j + n_wave] += dphi_peak * pulse[j] * symbols[n_sym - 1];
    }
    for (k = 0; k < n_wave; ++k) {
        signal[k] = sinf(phi);
        phi = fmodf(phi + dphi[k + n_spsym], 2.0f * (float)M_PI);
    }
    for (i = 0; i < n_ramp; ++i) {
        float env = (1.0f - cosf(2.0f * (float)M_PI * i / (2 * n_ramp))) / 2.0f;
        signal[i] *= env;
        signal[n_wave - 1 - i] *= env;
    }
    free(dphi);
    free(pulse);
    return 1;
}

int main(int argc, char** argv)
{
    const int sample_rate = 12000;
    const int num_tones = FT8_NN;
    const float audio_frequency = (argc > 3) ? (float)atof(argv[3]) : 1000.0f;
    const int num_samples = (int)(0.5f + num_tones * FT8_SYMBOL_PERIOD * sample_rate);
    const int total_samples = (int)(FT8_SLOT_TIME * sample_rate);
    const int silence = (total_samples - num_samples) / 2;
    ftx_message_t message;
    uint8_t tones[FT8_NN];
    float* signal;
    if (argc < 3) {
        fprintf(stderr, "usage: aetv_gen_ft8 MESSAGE WAV_FILE [AUDIO_HZ]\n");
        return 2;
    }
    if (ftx_message_encode(&message, NULL, argv[1]) != FTX_MESSAGE_RC_OK) {
        fprintf(stderr, "cannot encode FT8 message: %s\n", argv[1]);
        return 3;
    }
    signal = (float*)calloc((size_t)total_samples, sizeof(float));
    if (signal == NULL) {
        fprintf(stderr, "cannot allocate waveform\n");
        return 4;
    }
    ft8_encode(message.payload, tones);
    if (!synth_gfsk(tones, num_tones, audio_frequency, FT8_SYMBOL_PERIOD,
                    sample_rate, signal + silence)) {
        free(signal);
        return 5;
    }
    if (save_wav(signal, total_samples, sample_rate, argv[2]) != 0) {
        free(signal);
        fprintf(stderr, "cannot save waveform: %s\n", argv[2]);
        return 6;
    }
    free(signal);
    return 0;
}
