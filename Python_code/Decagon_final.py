import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load and group the datasets
def load_grouped_data(filenames):
    data_frames = []
    for file in filenames:
        df = pd.read_csv(file).groupby(['s [mm]', 'w [mm]', 'w2 [mm]']).agg({
            'Freq [GHz]': list,
            'dB(S(FloquetPort2:1,FloquetPort1:1)) []': list
        }).reset_index()
        data_frames.append(df)
    return pd.concat(data_frames, axis=0)

filenames = [f'New Data/{x}_ds.csv' for x in ['4p0', '4p5', '5p0', '5p5', '6p0', '6p5', '7p0', '7p5', '8p0']]
grouped_data = load_grouped_data(filenames)

# Prepare inputs (L1, L2, L3) and outputs (S21 values in dB)
X = grouped_data[['s [mm]', 'w [mm]', 'w2 [mm]']].values
y_db = np.array(grouped_data['dB(S(FloquetPort2:1,FloquetPort1:1)) []'].tolist())

# Normalize input and output data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y_db = scaler_y.fit_transform(y_db)

# Ensure consistent 8:1:1 split ratio
X_train, X_temp, y_train, y_temp = train_test_split(X, y_db, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the Forward Prediction Network (FPN) for dB values
def build_fpn():
    model = Sequential([
        Dense(64, input_dim=3),
        LeakyReLU(negative_slope=0.05),
        Dropout(0.2),  # Adding dropout
        Dense(128),
        LeakyReLU(negative_slope=0.05),
        Dropout(0.2),
        Dense(256),
        LeakyReLU(negative_slope=0.05),
        Dropout(0.2),
        Dense(128),
        LeakyReLU(negative_slope=0.05),
        Dense(71)  # Output: 71 neurons for S21 values in dB
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model

# Train the FPN model on dB values with early stopping and learning rate scheduler
fpn_model = build_fpn()
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

history = fpn_model.fit(X_train, y_train, epochs=200, batch_size=32, 
                        validation_data=(X_val, y_val), 
                        callbacks=[early_stopping, reduce_lr])

fpn_model.save('fpn_model_db.keras')

# Plot the training and validation loss to visually inspect convergence
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss for FPN Model')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.yscale('log')  # Log scale to better visualize convergence
plt.legend()
plt.grid(True)
plt.show()

# Define the Improved Particle Swarm Optimization (IPSO) with adaptive inertia weight and convergence threshold
class IPSO:
    def __init__(self, model, num_particles=15, max_iter=200, w_max=0.9, w_min=0.4, c1=2, c2_start=2, c2_end=0.1, wc=50, dimension=3, convergence_threshold=1e-4):
        self.model = model
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2_start = c2_start
        self.c2_end = c2_end
        self.wc = wc
        self.dimension = dimension
        self.convergence_threshold = convergence_threshold

        # Initialize particles and velocities within specified bounds
        self.particles = np.random.uniform([4.0, 0.2, 0.2], [8.0, 1.0, 1.0], (num_particles, dimension))
        self.velocities = np.zeros((num_particles, dimension))
        self.pbest = np.copy(self.particles)
        self.pbest_fitness = np.full(num_particles, float('inf'))
        self.gbest = None
        self.gbest_fitness = float('inf')
        self.fitness_values = []

    def fitness_function(self, particle, frequencies):
        s21_pred_db = self.model.predict(particle.reshape(1, -1))[0]
        interp_pred = interp1d(frequencies, s21_pred_db, kind='linear', fill_value="extrapolate")
        s21_at_targets = interp_pred([4.5, 6, 8.5, 10])

        # Fitness based on paper's formula
        fitness = ((s21_at_targets[0] + 15) ** 2 +
                   (s21_at_targets[1] + 15) ** 2 +
                   (s21_at_targets[2] + 3) ** 2 +
                   (s21_at_targets[3] + 3) ** 2)

        return fitness

    def calculate_inertia_weight(self, fitness, f_min, f_average):
        # Adaptive inertia weight calculation with convergence threshold
        if abs(self.gbest_fitness - fitness) < self.convergence_threshold:
            return self.w_min  # Converge faster with minimum inertia
        elif fitness <= f_average:
            return self.w_min + (fitness - f_min) / (f_average - f_min) * (self.w_max - self.w_min)
        else:
            return self.w_min

    def optimize(self, frequencies):
        for t in range(self.max_iter):
            fitness_scores = np.array([self.fitness_function(p, frequencies) for p in self.particles])
            f_min = np.min(fitness_scores)
            f_average = np.mean(fitness_scores)

            for i in range(self.num_particles):
                fitness = fitness_scores[i]
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest[i] = self.particles[i]
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest = self.particles[i]

            c2 = self.c2_end * (self.c2_start / self.c2_end) ** (1 / (1 + self.wc * (t / self.max_iter)))

            for i in range(self.num_particles):
                w = self.calculate_inertia_weight(fitness_scores[i], f_min, f_average)
                r1, r2 = np.random.rand(), np.random.rand()

                self.velocities[i] = (w * self.velocities[i]
                                      + self.c1 * r1 * (self.pbest[i] - self.particles[i])
                                      + c2 * r2 * (self.gbest - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], [4.0, 0.2, 0.2], [8.0, 1.0, 1.0])

            self.fitness_values.append(self.gbest_fitness)
            print(f"Iteration {t + 1} - Global Best Fitness: {self.gbest_fitness}")

        return self.gbest, self.gbest_fitness

# Initialize IPSO and run optimization
frequencies = grouped_data['Freq [GHz]'].iloc[0]
fpn_model = load_model('fpn_model_db.keras')

ipso = IPSO(model=fpn_model, num_particles=15, max_iter=200)
best_solution, best_fitness = ipso.optimize(frequencies)

# Print optimal solution
print("Optimal FSS parameters (L1, L2, L3):", best_solution)
print("Best Fitness Score:", best_fitness)

# Predict and plot S21 in dB using the best parameters
predicted_s21_db = fpn_model.predict(best_solution.reshape(1, -1))[0]

# Plot the actual (target) and predicted S21 values in dB
plt.figure(figsize=(10, 6))
plt.plot(frequencies, predicted_s21_db, label="Predicted S21 (dB)", color='blue')
plt.title("Predicted S21 vs Frequency")
plt.xlabel("Frequency (GHz)")
plt.ylabel("S21 (dB)")
plt.legend()
plt.grid(True)
plt.show()
