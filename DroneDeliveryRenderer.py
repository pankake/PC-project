
from matplotlib import pyplot as plt

import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

FONT_SIZE_M = 10
FONT_SIZE_S = 8

# costanti per i colori
ULTRA_RED = '#FC6C85'
SALOMON_PINK = '#FC94A1'
LIGHT_RED = '#FFCCC9'
TEA_GREEN = '#CFFDCC'
MENTHAL = '#B0F5AB'
LIGHT_GREEN = '#90EF90'
FULLY_CHARGED = '#90EE90'

# determina il colore basato sul valore di charging_timer
def get_drone_color(charging_timer):
    if charging_timer >= 6:
        return ULTRA_RED  # valori >= 6
    elif charging_timer >= 5:
        return SALOMON_PINK  # valori 5 <= charging_timer < 6
    elif charging_timer >= 4:
        return LIGHT_RED  # valori 4 <= charging_timer < 5
    elif charging_timer >= 3:
        return TEA_GREEN  # valori 3 <= charging_timer < 4
    elif charging_timer >= 2:
        return MENTHAL  # valori 2 <= charging_timer < 3
    elif charging_timer >= 1:
        return LIGHT_GREEN  # valori 1 <= charging_timer < 2
    else:
        return FULLY_CHARGED  # valori < 1


class DroneDeliveryRenderer:
    @staticmethod
    def render(env):
        # crea una griglia vuota
        grid = np.zeros(env.grid_size)

        # assegna il valore 1 ad ogni delivery point
        for delivery_point in env.target_delivery_points:
            if delivery_point is not None:
                x, y = delivery_point
                grid[x, y] = 1

        # assegna il valore 2 ad ogni stazione di ricarica
        for charging_station in env.CHARGING_STATIONS:
            grid[charging_station[0], charging_station[1]] = 2

        # assegna il valore 3 al magazzino
        warehouse = env.WAREHOUSE
        grid[warehouse[0], warehouse[1]] = 3

        # lista per rappresentare le zone di maltempo sotto forma di rettangoli
        if not hasattr(env, 'weather_zone_patches'):
            env.weather_zone_patches = []

        # prima di disegnare rimuove le vecchie zone di maltempo
        for patch in env.weather_zone_patches:
            patch.remove()
        env.weather_zone_patches.clear()

        # colore blu per le zone di maltempo con trasparenza del 40%
        weather_zone_color = (0.5, 0.8, 1.0, 0.4)

        # acquisisce i dati per le zone di maltempo attive
        for weather_zone in env.weather_zones:
            (y, x), (width, height), _ = weather_zone.values()
            rect = plt.Rectangle((y - 0.5, x - 0.5), width, height, color=weather_zone_color, lw=0)
            env.ax.add_patch(rect)
            env.weather_zone_patches.append(rect)

        # imposta un valore di default temporaneo per la posizione dei droni
        drone_positions = np.full(env.grid_size, -1)

        # aggiunge i droni sulla griglia e gestisce i colori
        for i, drone_state in enumerate(env.drone_states):
            x, y, battery_level, _, _, _, _, has_package, charging_timer, _, _, _ = drone_state

            # verifica se il drone è su una stazione di ricarica
            if (x, y) in env.CHARGING_STATIONS:
                drone_color = get_drone_color(charging_timer)
                env.ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color=drone_color, lw=0))
            else:
                drone_positions[x, y] = 4 + i  # ogni drone ha un valore unico


        # rappresenta i droni sulla griglia
        for x in range(env.grid_size[0]):
            for y in range(env.grid_size[1]):
                if drone_positions[x, y] != -1:
                    if grid[x, y] == 0:
                        grid[x, y] = drone_positions[x, y]

        # imposta i colori per tutti gli elementi
        cmap = ListedColormap(['#D3D3D3', 'yellow', '#90EE90', '#D2B48C', 'azure', 'azure', 'azure', '#4682B4'])
        bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        norm = BoundaryNorm(bounds, cmap.N)

        # controlla che ax e canvas siano inizializzati
        if env.ax is None or env.canvas is None:
            print("Error: canvas or ax not initialized correctly")
            return

        # inizializza l'immagine solo la prima volta
        if env.im is None:
            env.im = env.ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
        else:
            env.im.set_data(grid)

        # calcola le percentuali di batteria
        max_battery_level = env.BATTERY_LEVELS  # env.BATTERY_LEVELS vale 20
        battery_percentages = [(drone_state[2] / max_battery_level) * 100 for drone_state in env.drone_states]

        # crea una stringa con i dettagli dei droni e le percentuali di batteria
        battery_levels_str = ", ".join(
            [f"Drone {i + 1}: {battery_percentage:.1f}%" for i, battery_percentage in enumerate(battery_percentages)]
        )

        # crea una stringa con il dettaglio delle consegne effettuate per ogni drone
        completed_deliveries_str = ", ".join(
            [f"Drone {i + 1}: {completed}" for i, completed in enumerate(env.deliveries_completed)]
        )

        # calcola il totale complessivo delle consegne completate
        total_deliveries = sum(env.deliveries_completed)

        # aggiorna il titolo con il totale delle consegne e i valori della batteria
        title = (f"Battery Percentages: {battery_levels_str}\n"
                 f"Deliveries Completed: {completed_deliveries_str} (Total: {total_deliveries}/{env.WAREHOUSE_ITEMS})")

        env.ax.set_title(title, fontsize=FONT_SIZE_M)

        # rimuove i numeri e le etichette degli assi x e y del disegno
        env.ax.set_xticks([])
        env.ax.set_yticks([])
        env.ax.set_xticklabels([])
        env.ax.set_yticklabels([])

        # rimuove le etichette dei droni a ogni passo
        for label in env.drone_labels:
            if label is not None:
                label.remove()
        env.drone_labels = [None] * len(env.drone_states)

        # ricava la posizione delle stazioni di ricarica
        charging_station_positions = {charging_station: i for i, charging_station in enumerate(env.CHARGING_STATIONS)}

        # rimuove le etichette dei droni se si trovano su una stazione di ricarica
        drone_positions_on_charging_stations = set()
        for i, drone_state in enumerate(env.drone_states):
            x, y, _, _, _, _, _, _, _, _, _, _ = drone_state
            if (x, y) in charging_station_positions:
                drone_positions_on_charging_stations.add((x, y))
                env.drone_labels[i] = None  # rimuove l'etichetta del drone che è sulla stazione di ricarica

        # aggiunge le nuove etichette per i droni
        for i, drone_state in enumerate(env.drone_states):
            x, y, battery_level, _, _, _, _, _, _, _, _, _ = drone_state
            if (x, y) not in drone_positions_on_charging_stations:
                # quando il drone si trova sul magazzino l'etichetta non viene renderizzata
                if (x, y) != warehouse:
                    env.drone_labels[i] = env.ax.text(y, x, f'D{i + 1}', ha='center', va='center', fontsize=FONT_SIZE_S,
                                                      color='black', fontweight='bold')

        # rimuove le etichette dei delivery point esistenti
        for label in env.delivery_points_labels:
            if label is not None:
                label.remove()
        env.delivery_points_labels = [None] * len(env.target_delivery_points)

        # aggiunge le nuove etichette ai punti di consegna
        for i, delivery_point in enumerate(env.target_delivery_points):
            if delivery_point is not None:
                x, y = delivery_point
                label = env.ax.text(y, x, f'DP{i + 1}', ha='center', va='center', fontsize=FONT_SIZE_S, color='black',
                                    fontweight='bold')
                env.delivery_points_labels.append(label)

        # label per il magazzino
        warehouse_label_text = f"Warehouse\n({env.num_objects})"
        drone_at_warehouse = next((f'D{i + 1}' for i, drone_state in enumerate(env.drone_states)
                                   if (drone_state[0], drone_state[1]) == warehouse), None)

        # quando il drone si trova sul magazzino
        if drone_at_warehouse:
            warehouse_label_text = f"Load\n({drone_at_warehouse})"

        if env.warehouse_label is None:
            env.warehouse_label = env.ax.text(warehouse[1], warehouse[0], warehouse_label_text,
                                              ha='center', va='center', fontsize=FONT_SIZE_S, color='black',
                                              fontweight='bold')
        else:
            env.warehouse_label.set_text(warehouse_label_text)

        # aggiunge le etichette per le stazioni di ricarica
        for i, charging_station in enumerate(env.CHARGING_STATIONS):
            x, y = charging_station
            # verifica se un drone si trova sulla stazione di ricarica
            drone_at_station = next((f'D{i + 1}' for i, drone_state in enumerate(env.drone_states)
                                     if (drone_state[0], drone_state[1]) == charging_station), None)

            if drone_at_station:
                # mostra l'etichetta aggiornata per la stazione di ricarica con il numero del drone
                label_text = f'Charge\n({drone_at_station})'
            else:
                # mostra etichetta normale per la stazione di ricarica
                label_text = f'R{i + 1}'

            if env.charging_stations_labels[i] is None:
                env.charging_stations_labels[i] = (
                    env.ax.text(y, x, label_text, ha='center', va='center', fontsize=FONT_SIZE_S, color='black',
                                fontweight='bold'))
            else:
                env.charging_stations_labels[i].set_text(label_text)

        # aggiorna il canvas per visualizzare i cambiamenti
        env.canvas.draw()
        env.root.update()