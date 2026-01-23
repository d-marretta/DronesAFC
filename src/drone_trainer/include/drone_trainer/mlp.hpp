#ifndef MLP_HPP
#define MLP_HPP

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>


const int POP_SIZE = 128;         // Number of "drones" to test per generation
const float SIGMA = 0.15f;        // Noise standard deviation
const float ALPHA = 0.01f;       // Learning rate
const float EPISODE_TIME = 12.0f; // Seconds per flight
const int INPUT_SIZE = 22;        // Dx, Dy, Dz, Vel X, Vel Y, Vel Z
const int HIDDEN_SIZE = 256;      // Neurons in hidden layer
const int OUTPUT_SIZE = 4;       // Linear X, Linear Y, Linear Z, Angular Z

const float START_X = 0.0f;
const float START_Y = 0.0f;
const float START_Z = 1.0f;

const float MAX_LIN_VEL_X = 3.0f;  // Max horizontal speed (m/s)
const float MAX_LIN_VEL_Y = 3.0f;
const float MAX_LIN_VEL_Z  = 1.0f;  // Max vertical speed (m/s)
const float MAX_ANG_VEL_Z  = 1.5f;  // Max rotational speed (rad/s)
const float MAX_DIST_RANGE = 30.0f; // Inputs beyond are clamped to 1.0
const float MAX_VEL_RANGE  = 3.0f;  // Inputs beyond are clamped to 1.0
const float MAX_LIDAR_DIST = 10.0f;

inline float tanh_activation(float x) { return std::tanh(x); }

struct MLP {
    std::vector<float> weights;

    // Initialize random weights
    MLP() {
        int count = (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE);
        weights.resize(count);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> d(0.0, 0.2);
        for(float &w : weights) w = d(gen);
    }

    // Forward
    std::vector<float> forward(const std::vector<float>& state, const std::vector<float>& noise = {}, float noise_scale = 0.0) {
        std::vector<float> w = weights;
        if (!noise.empty()) {
            for(size_t i=0; i<w.size(); i++) w[i] += noise[i] * noise_scale;
        }

        // Input -> Hidden
        std::vector<float> hidden(HIDDEN_SIZE, 0.0f);
        int idx = 0;
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            float sum = 0.0f;
            for (int i = 0; i < INPUT_SIZE; i++) {
                sum += state[i] * w[idx++];
            }
            hidden[h] = tanh_activation(sum);
        }

        // Hidden -> Output 
        std::vector<float> outputs(OUTPUT_SIZE, 0.0f);
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            float sum = 0.0f;
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                sum += hidden[h] * w[idx++];
            }
            outputs[o] = tanh_activation(sum); 
        }
        return outputs;
    }

    void save(const std::string& filename) {
        std::ofstream file(filename);
        if (file.is_open()) {
            for (float w : weights) {
                file << w << "\n";
            }
            file.close();
        }
    }
    
    int get_weight_count() { return weights.size(); }
};

#endif