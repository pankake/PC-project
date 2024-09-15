import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

from DroneDeliveryEnvironment import DroneDeliveryEnvironment
from DroneDeliveryRenderer import DroneDeliveryRenderer


class DroneDeliverySimulation:
    def __init__(self, env, root):
        self.env = env  # inizializza l'ambiente
        self.root = root # inizializza l'istanza dell'interfaccia grafica
        self.states = env.reset()  # inizializza lo stato dei droni
        self.done = [False, False, False]  # stato di completamento per ciascun drone
        self.env.epsilon = 0  # impostato a zero per annullare l'esplorazione durante la simulazione

    def __step_simulation(self):
        if not all(self.done):
            for i in range(len(self.states)):
                if not self.done[i]:  # controlla se il drone ha già terminato
                    action = self.env.choose_action(self.states[i])  # azione basata sullo stato del drone i-esimo
                    next_state, reward, done = self.env.step(i, action)  # esegue uno step per il drone i-esimo
                    self.states[i] = next_state
                    self.done[i] = done

                    if self.states[i][2] == 0:  # se la batteria è esaurita
                        print(f"Battery depleted for Drone {i+1}. Ending simulation for this drone. State: {self.states[i]}")
                        self.done[i] = True

            # aggiorna la GUI e richiama il prossimo step
            DroneDeliveryRenderer.render(self.env)
            self.root.after(300, self.__step_simulation)
        else:
            print("Simulation finished.")

    def run(self):
        self.root.after(0, self.__step_simulation)

def main():
    grid_size = (7, 7)

    root = tk.Tk()
    root.title("Drone Delivery Simulation")

    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # crea l'istanza dell'ambiente
    env = DroneDeliveryEnvironment(grid_size, root=root, canvas=canvas, ax=ax, training_mode=False)

    # carica la q-table
    try:
        env.Q_table = np.load("q_table.npy")
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("Error: q-table not found.")
        return

    # inizializza la simulazione con l'ambiente
    simulation = DroneDeliverySimulation(env, root)

    print("Starting simulation...")
    # avvia la simulazione
    simulation.run()
    # avvia l'interfaccia grafica
    root.mainloop()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
