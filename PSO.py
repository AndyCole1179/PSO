import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image

np.set_printoptions(formatter={'float_kind': lambda x: '{0:.8f}'.format(x)})
# definisi fungsi f(x, y) = (2 - x)^2 + 10(y^2 - x)^2
def objective_function(position):
    x, y = position
    return (2 - x)**2 + 10 * (y**2 - x)**2

class Particle:
    def __init__(self, dim, batas):
        self.position = np.random.uniform(batas[0], batas[1], dim)  # inisiasi posisi acak 
        self.velocity = np.random.uniform(-1, 1, dim)  # inisiasi kecepatan acak
        self.best_position = np.copy(self.position)  # posisi terbaik partikel
        self.best_value = objective_function(self.position)  # nilai terbaik partikel

class PSO:
    def __init__(self, objective_function, dim, batas, num_particles=10, max_iter=100, c1=1, c2=0.5):
        self.obj_func = objective_function 
        self.dim = dim  # dimensi 
        self.batas = batas 
        self.num_particles = num_particles  # jumlah partikel
        self.max_iter = max_iter  # jumlah iterasi maksimal
        self.c1 = c1  # posisi terbaik individu
        self.c2 = c2  # posisi terbaik global
        self.particles = [Particle(dim, batas) for _ in range(num_particles)]  # inisialisasi partikel
        self.global_best_position = None  # posisi terbaik global
        self.global_best_value = float("inf")  # nilai terbaik global
        self.iteration_logs = []  # log setiap iterasi

    def initialize_random_positions(self):
        for i in range(self.num_particles):
            self.particles[i].position = np.random.uniform(self.batas[0], self.batas[1], self.dim)

    def optimize(self):
        self.initialize_random_positions()  # inisialisasi posisi partikel secara acak

        for iteration in range(self.max_iter):
            iteration_log = {"positions": [], "velocities": [], "PBest": [], "gBest": None}

            for particle in self.particles:
                value = self.obj_func(particle.position)  # evaluasi fungsi objektif pada posisi partikel
                if value < particle.best_value:  # jika ditemukan nilai yang lebih baik
                    particle.best_value = value  # update nilai terbaik
                    particle.best_position = np.copy(particle.position)  # update posisi terbaik

                if value < self.global_best_value:  # jika ditemukan nilai yang lebih baik secara global
                    self.global_best_value = value  # update nilai terbaik global
                    self.global_best_position = np.copy(particle.position)  # update posisi terbaik global

            for particle in self.particles: # update kecepatan dan posisi partikel
                iteration_log["positions"].append(np.copy(particle.position))
                iteration_log["velocities"].append(np.copy(particle.velocity))
                iteration_log["PBest"].append(np.copy(particle.best_position))

                r1 = np.random.rand(self.dim)  # angka acak antara 0 dan 1
                r2 = np.random.rand(self.dim)  # angka acak antara 0 dan 1
                inertia = 0.5  # faktor inersia (pengaruh kecepatan sebelumnya)
                cognitive = self.c1  # koefisien untuk posisi terbaik pribadi
                social = self.c2  # koefisien untuk posisi terbaik global

                # update kecepatan partikel
                particle.velocity = inertia * particle.velocity + cognitive * r1 * (particle.best_position - particle.position) + social * r2 * (self.global_best_position - particle.position)

                # update posisi partikel
                particle.position = particle.position + particle.velocity
                # memastikan partikel tetap berada dalam batas pencarian
                particle.position = np.clip(particle.position, self.batas[0], self.batas[1])

            iteration_log["gBest"] = {
                "position": np.copy(self.global_best_position),
                "value": self.global_best_value
            }
            # Simpan log iterasi
            self.iteration_logs.append(iteration_log)

        return self.global_best_position, self.global_best_value
    
# Fungsi untuk membuat direktori dan menyimpan gambar
def plot_pso_logs(iteration_logs, output_dir="pngs"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    num_digits = len(str(max_iter))  # Menghitung jumlah digit maksimum

    for i, log in enumerate(iteration_logs):
        positions = np.array(log["positions"])  # Ambil posisi partikel
        gBest = np.array(log["gBest"]["position"])  # Ambil global best
        plt.figure(figsize=(8, 6))
        plt.scatter(positions[:, 0], positions[:, 1], color='dodgerblue', label=f'Particles - Iteration {i+1}')
        plt.scatter(gBest[0], gBest[1], color='red', marker='x', s=100, label='Global Best')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'PSO Iteration {i+1}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"iteration_{i+1:0{num_digits}d}.png"))
        plt.close()

# Fungsi untuk membuat GIF
def create_gif_from_images(output_dir, gif_name="pso_animation.gif"):
    images = []
    for file_name in sorted(os.listdir(output_dir)):
        if file_name.endswith(".png"):
            file_path = os.path.join(output_dir, file_name)
            images.append(Image.open(file_path))
    gif_path = os.path.join(output_dir, gif_name)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
    print(f"Successfully saved gif: {gif_path}")

# Main program
if __name__ == "__main__":
    dim = 2
    batas = [-10, 10]
    max_iter = 100
    pso = PSO(objective_function, dim, batas, num_particles=10, max_iter=max_iter, c1=1, c2=0.5)
    best_position, best_value = pso.optimize()

    # Menyimpan visualisasi setiap iterasi
    plot_pso_logs(pso.iteration_logs, "pngs")

    # Membuat GIF dari gambar yang disimpan
    create_gif_from_images("pngs")

    # Menampilkan hasil log untuk setiap iterasi
    for i, log in enumerate(pso.iteration_logs):
        print(f"\n--- Iterasi {i + 1} ---")
        print(f"gBest: Position = {log['gBest']['position']}, Value = {log['gBest']['value']}\n")
    
        print(f"{'Particle':<10}{'Position (x, y)':<25}{'Velocity (v_x, v_y)':<25}{'PBest (x, y)':<25}")
        print("-" * 80)
        for j, (pos, vel, pbest) in enumerate(zip(log["positions"], log["velocities"], log["PBest"])):
            print(f"{j + 1:<10}{str(pos):<25}{str(vel):<25}{str(pbest):<25}")

    print("Best Position (x, y):", best_position)
    print("Best Value:", best_value)