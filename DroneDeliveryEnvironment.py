
import numpy as np
import random

from collections import deque
from ActionType import ActionType


class DroneDeliveryEnvironment:
    def __init__(self, grid_size, epsilon=0.5, root=None, canvas=None, ax=None,
                 training_mode=True):
        self.grid_size = grid_size
        self.WAREHOUSE_ITEMS = 20
        self.BATTERY_LEVELS = 40
        self.LOW_BATTERY_THRESHOLD = 20
        self.CHARGING_STATIONS = [(3, 3), (6, 0), (0, 6)]
        self.WAREHOUSE = (grid_size[0] // 2, grid_size[1] - 1)
        self.CHARGING_TIME = 7

        self.drone_states = [
            (3, 3, self.BATTERY_LEVELS - 1, 0, 0, 0, 0, False, 0, ActionType.SKIP.value, 0, []),
            (6, 0, self.BATTERY_LEVELS - 1, 0, 0, 0, 0, False, 0, ActionType.SKIP.value, 0, []),
            (0, 6, self.BATTERY_LEVELS - 1, 0, 0, 0, 0, False, 0, ActionType.SKIP.value, 0, [])
        ]

        self.actions = [ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT,
                        ActionType.SKIP, ActionType.CIRCUMNAVIGATE]

        self.num_actions = len(self.actions)

        self.Q_table = np.zeros((2, 2, 2, 2, 5, 2, self.num_actions))

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = epsilon
        self.num_objects = self.WAREHOUSE_ITEMS
        self.deliveries_completed = [0, 0, 0]
        self.training_mode = training_mode
        self.root = root
        self.canvas = canvas
        self.ax = ax
        self.im = None
        self.drone_labels = [None, None, None]
        self.warehouse_label = None
        self.charging_stations_labels = [None, None, None]
        self.delivery_points_labels = [None, None, None]
        self.count = 0
        self.target_delivery_points = [None, None, None]
        self.weather_zones = []  # lista per memorizzare le zone di maltempo
        self.weather_frequency = 20  # frequenza con cui appaiono le zone di maltempo (in numero di step)
        self.weather_lifetime = 20  # durata delle zone di maltempo (in step)
        self.weather_zone_patches = []

        # memorizza le coordinate degli elementi
        self.elements_coordinates = {
            'drones': [state[:2] for state in self.drone_states],
            'delivery_points': [None] * len(self.drone_states),
            'warehouse': [self.WAREHOUSE],
            'charging_stations': self.CHARGING_STATIONS
        }

    def reset(self):
        self.drone_states = [
            (3, 3, self.BATTERY_LEVELS - 1, 0, 0, 0, 0, False, 0, ActionType.SKIP.value, 0, []),
            (6, 0, self.BATTERY_LEVELS - 1, 0, 0, 0, 0, False, 0, ActionType.SKIP.value, 0, []),
            (0, 6, self.BATTERY_LEVELS - 1, 0, 0, 0, 0, False, 0, ActionType.SKIP.value, 0, [])
        ]
        self.target_delivery_points = [None, None, None]
        self.deliveries_completed = [0, 0, 0]
        self.num_objects = self.WAREHOUSE_ITEMS

        return self.drone_states

    def choose_action(self, state):
        _, _, battery_level, obstacle_up, obstacle_down, obstacle_left, obstacle_right, _, _, relative_target_position, circumnavigate, _, = state
        obstacle_state = (obstacle_up, obstacle_down, obstacle_left, obstacle_right)
        circumnavigate_state = 1 if circumnavigate else 0

        q_values = self.Q_table[
            obstacle_state[0], obstacle_state[1], obstacle_state[2], obstacle_state[3], relative_target_position, circumnavigate_state]
        if random.uniform(0, 1) < self.epsilon or np.sum(q_values) == 0:
            #if np.sum(q_values) == 0:
                # self.count += 1
                # print("random action: " + str(self.count))
            return random.choice(range(self.num_actions))
        else:
            return np.argmax(q_values)

    def update_q_table(self, state, action, reward, next_state):
        _, _, _, obstacle_up, obstacle_down, obstacle_left, obstacle_right, _, _, relative_target_position, circumnavigate, _ = state
        _, _, _, next_obstacle_up, next_obstacle_down, next_obstacle_left, next_obstacle_right, _, _, next_relative_target_position, next_circumnavigate, _ = next_state

        # valori del vecchio stato
        obstacle_state = (obstacle_up, obstacle_down, obstacle_left, obstacle_right)
        next_obstacle_state = (next_obstacle_up, next_obstacle_down, next_obstacle_left, next_obstacle_right)

        circumnavigate_state = 1 if circumnavigate else 0
        next_circumnavigate_state = 1 if next_circumnavigate else 0

        # ricava la migliore azione che si può scegliere nel nuovo stato
        best_next_action = np.argmax(self.Q_table[
                                         next_obstacle_state[0], next_obstacle_state[1], next_obstacle_state[2],
                                         next_obstacle_state[3], next_relative_target_position, next_circumnavigate_state])

        # calcola il target che rappresenta il valore atteso della ricompensa futura:
        td_target = reward + self.gamma * self.Q_table[
            next_obstacle_state[0], next_obstacle_state[1], next_obstacle_state[2], next_obstacle_state[
                3], next_relative_target_position, next_circumnavigate_state, best_next_action]

        # calcola l'errore: quello che è venuto in meno rispetto al valore atteso
        td_error = td_target - self.Q_table[
            obstacle_state[0], obstacle_state[1], obstacle_state[2], obstacle_state[
                3], relative_target_position, circumnavigate_state, action]

        # aggiorna la q-table per l'azione corrente, migliorando le scelte future
        self.Q_table[
            obstacle_state[0], obstacle_state[1], obstacle_state[2], obstacle_state[
                3], relative_target_position, circumnavigate_state, action] += self.alpha * td_error

    def step(self, drone_index, action):
        state = self.drone_states[drone_index]
        (y, x, battery_level, obstacle_up, obstacle_down, obstacle_left, obstacle_right, has_package,
         charging_timer, relative_tgt, circumnavigate, circumnavigation_path) = state

        self.__update_weather_zones()

        reward = 0
        done = False

        new_y = y
        new_x = x

        # se il drone sta caricando, decrementa il timer e salta le operazioni
        if charging_timer > 0:
            charging_timer -= 1

            # aggiorna lo stato del drone con il timer decrementato
            self.drone_states[drone_index] = (
                y, x, battery_level, obstacle_up, obstacle_down, obstacle_left, obstacle_right,
                has_package, charging_timer, relative_tgt, circumnavigate, circumnavigation_path)

            # ritorna lo stato aggiornato del drone
            return self.drone_states[drone_index], 0, False  # nessuna ricompensa durante la ricarica

        # il drone non ha più compiti da svolgere
        elif self.num_objects == 0 and not has_package and (y, x) == self.CHARGING_STATIONS[drone_index]:
            return self.drone_states[drone_index], 0, True

        try:
            action_type = ActionType(action)
        except ValueError as e:
            print(f"Invalid action {action} for drone {drone_index}: {e}")
            action_type = ActionType.SKIP  # azione di default

        new_battery_level = max(0, battery_level - 1)

        if circumnavigate:
            if action_type == ActionType.CIRCUMNAVIGATE:

                if new_battery_level < self.LOW_BATTERY_THRESHOLD:
                    # azzera il percorso di circumnavigazione se la batteria è bassa
                    circumnavigation_path = []

                if not circumnavigation_path:
                    target = self.__determine_target(new_battery_level, has_package, drone_index)
                    circumnavigation_path = self.__calculate_weather_circumnavigation_path((y, x), target, drone_index)

                if circumnavigation_path:
                    new_y, new_x = circumnavigation_path.pop(0)  # pop della prossima cella del percorso
                else:
                    # se la lista è vuota o non c'è necessità di circumnavigare, mantiene la posizione corrente
                    new_y, new_x = y, x

                reward += 50  # grande ricompensa per l'azione corretta
            else:
                reward -= 50  # penalizza perché ha scelto un'azione che non è circumnavigazione
        else:
            if action_type == ActionType.CIRCUMNAVIGATE:
                reward -= 30  # penalità perché ha scelto circumnavigazione quando non è necessario

        if action_type != ActionType.CIRCUMNAVIGATE:
            circumnavigation_path = []

        if action_type == ActionType.UP:
            if y > 0:  # muove su se non è già al bordo
                new_y -= 1

            if obstacle_up:
                reward -= 20
            elif relative_tgt == ActionType.UP.value:
                reward += 10  # ricompensa per il movimento corretto
            else:
                reward -= 5  # penalità per tentare di uscire dal bordo

        elif action_type == ActionType.DOWN:
            if y < self.grid_size[1] - 1:  # muove giù se non è al bordo inferiore
                new_y += 1

            if obstacle_down:
                reward -= 20
            elif relative_tgt == ActionType.DOWN.value:
                reward += 10
            else:
                reward -= 5

        elif action_type == ActionType.LEFT:
            if x > 0:  # muove a sinistra se non è già al bordo sinistro
                new_x -= 1

            if obstacle_left:
                reward -= 20
            elif relative_tgt == ActionType.LEFT.value:
                reward += 10
            else:
                reward -= 5

        elif action_type == ActionType.RIGHT:
            if x < self.grid_size[0] - 1:  # muove a destra se non è al bordo destro
                new_x += 1

            if obstacle_right:
                reward -= 20
            elif relative_tgt == ActionType.RIGHT.value:
                reward += 10
            else:
                reward -= 5

        # evita di collidere: impedisce di aggiornare la posizione in una cella occupata
        new_y, new_x = self.__check_collision(drone_index, new_y, new_x, y, x)

        # il percorso viene ricalcolato in caso di collisione o se è presente un ostacolo sul percorso
        if circumnavigate and (new_y, new_x) == (y, x) or (new_y, new_x) in self.__get_obstacles(drone_index):
            # ricalcola il percorso se la cella è diventata occupata
            start_position = (y, x)
            target = self.__determine_target(new_battery_level, has_package, drone_index)
            path = self.__calculate_weather_circumnavigation_path(start_position, target, drone_index)

            if path:
                # aggiorna la lista delle celle da percorrere
                circumnavigation_path = path
                # prende la nuova cella dal percorso e aggiorna la posizione
                new_y, new_x = circumnavigation_path.pop(0) if circumnavigation_path else (y, x)

        # gestione della ricarica della batteria
        new_battery_level, charging_timer = self.__recharge_battery(new_y, new_x, new_battery_level, drone_index)

        # gestione della consegna del pacco
        has_package = self.__deliver_package(new_y, new_x, has_package, drone_index)
        # gestione del ritiro del pacco
        has_package = self.__pick_up_package(new_y, new_x, has_package, drone_index)

        target = self.__determine_target(new_battery_level, has_package, drone_index)

        # determina la posizione relativa del target rispetto al drone
        relative_target_position = self.__get_relative_target_position((new_y, new_x), target)

        # rilevamento degli ostacoli
        obstacle_up, obstacle_down, obstacle_left, obstacle_right = self.__detect_obstacles(
            (new_y, new_x, new_battery_level, has_package), drone_index)

        circumnavigate = True if circumnavigation_path else self.__needs_circumnavigation((y, x), target, relative_target_position,
                                                                                          obstacle_up, obstacle_down, obstacle_left, obstacle_right)

        # aggiorna la posizione del drone nella lista delle coordinate
        self.elements_coordinates['drones'][drone_index] = (new_y, new_x)

        new_battery_level = self.__decrement_battery_due_to_weather(new_y, new_x, new_battery_level)

        new_state = (new_y, new_x, new_battery_level, obstacle_up, obstacle_down,
                     obstacle_left, obstacle_right, has_package, charging_timer, relative_target_position, circumnavigate, circumnavigation_path)

        self.drone_states[drone_index] = new_state

        return new_state, reward, done

    def __calculate_circumnavigation_path(self, start_position, target_position, drone_index):
        # ricava la lista degli ostacoli
        obstacles = self.__get_obstacles(drone_index)

        # crea una coda a doppia estremità e la inizializza con la posizione di partenza e un percorso vuoto
        # ogni elemento della coda è una tupla che contiene la posizione corrente e il percorso esplorato
        queue = deque([(start_position, [])])

        # set per tracciare le posizioni visitate
        visited = set()
        visited.add(start_position)

        # direzioni possibili
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # finché la coda non è vuota continua a esplorare il percorso
        while queue:
            # estrae il primo elemento della coda, posizione corrente e percorso associato
            (current_position, path) = queue.popleft()
            y, x = current_position

            # se ha raggiunto il target restituisce il percorso
            if current_position == target_position:
                return path

            # esplora tutte le celle adiacenti
            for dy, dx in directions:
                new_y, new_x = y + dy, x + dx
                new_position = (new_y, new_x)

                # verifica se la nuova posizione è valida e non è un ostacolo
                if (0 <= new_y < self.grid_size[0] and 0 <= new_x < self.grid_size[1]  # Check dentro i limiti
                        and new_position not in obstacles and new_position not in visited):
                    visited.add(new_position)
                    queue.append((new_position, path + [new_position]))

        # ritorna una lista vuota se non c'è un percorso disponibile
        return []

    def __needs_circumnavigation(self, current_pos, target, relative_target, obstacle_up, obstacle_down, obstacle_left, obstacle_right):
        # c'è un ostacolo tra il drone e il target, quindi circumnaviga
        if (relative_target == ActionType.UP.value and obstacle_up) or \
                (relative_target == ActionType.DOWN.value and obstacle_down) or \
                (relative_target == ActionType.LEFT.value and obstacle_left) or \
                (relative_target == ActionType.RIGHT.value and obstacle_right):
            return True

        # il target è in alto o in basso, circumnaviga eventuali ostacoli nelle direzioni orizzontali
        if relative_target == ActionType.UP.value or relative_target == ActionType.DOWN.value:
            if obstacle_left or obstacle_right:
                return True

        # il target è a sinistra o a destra, controlla raggira eventuali ostacoli nelle direzioni verticali
        if relative_target == ActionType.LEFT.value or relative_target == ActionType.RIGHT.value:
            if obstacle_up or obstacle_down:
                return True

        # verifica se ci sono zone con maltempo da circumnavigare
        return self.__detect_weather_zones(current_pos, target, relative_target)

    def __get_relative_target_position(self, drone_position, target_position):
        # coordinate del drone e del target
        y_drone, x_drone = drone_position
        y_target, x_target = target_position

        # calcolo delle differenze tra le coordinate
        delta_x = x_target - x_drone
        delta_y = y_target - y_drone

        # se il target è più distante verticalmente, viene data priorità al movimento verticale
        if abs(delta_y) > abs(delta_x):
            if delta_y < 0:
                return ActionType.UP.value  # su
            elif delta_y > 0:
                return ActionType.DOWN.value  # giù

        # se il target è più distante orizzontalmente o ha la stessa distanza rispetto a y, si muove orizzontalmente
        if abs(delta_x) >= abs(delta_y):
            if delta_x < 0:
                return ActionType.LEFT.value  # sx
            elif delta_x > 0:
                return ActionType.RIGHT.value  # dx

        # se il target è esattamente sulla stessa posizione del drone
        return ActionType.SKIP.value

    def __recharge_battery(self, y, x, new_battery_level, drone_index):
        if ((y, x) == self.CHARGING_STATIONS[drone_index]
                and new_battery_level < self.BATTERY_LEVELS - self.LOW_BATTERY_THRESHOLD):

            # calcola il livello di batteria mancante per raggiungere il massimo
            battery_needed = self.BATTERY_LEVELS - new_battery_level

            # calcola il tempo di ricarica in relazione alla batteria rimasta
            charging_timer = (battery_needed / self.BATTERY_LEVELS) * self.CHARGING_TIME

            # ripristina il massimo livello di batteria
            new_battery_level = self.BATTERY_LEVELS

            # restituisce il tempo necessario per la ricarica se non si sta addestrando
            if not self.training_mode:
                return new_battery_level, charging_timer

        return new_battery_level, 0

    def __get_obstacles(self, drone_index):
        obstacles = set()

        # aggiunge i delivery points degli altri droni come ostacoli
        for i, dp in enumerate(self.elements_coordinates['delivery_points']):
            if i != drone_index:
                obstacles.add(dp)

        # aggiunge le charging stations degli altri droni come ostacoli
        for i, cs in enumerate(self.elements_coordinates['charging_stations']):
            if i != drone_index:
                obstacles.add(cs)

        # aggiunge la propria charging station solo se il livello della batteria è maggiore della soglia
        current_drone_state = self.drone_states[drone_index]
        y, x, battery_level, _, _, _, _, has_package, _, _, _, _ = current_drone_state

        if battery_level > self.LOW_BATTERY_THRESHOLD and self.num_objects > 0:
            own_charging_station = self.elements_coordinates['charging_stations'][drone_index]
            obstacles.add(own_charging_station)

        # aggiunge le coordinate degli altri droni come ostacoli
        for i, drone_coords in enumerate(self.elements_coordinates['drones']):
            if i != drone_index:
                obstacles.add(drone_coords)

        if has_package:
            obstacles.add(self.WAREHOUSE)

        return obstacles

    def __deliver_package(self, y, x, has_package, drone_index):
        # verifica se il drone ha un pacco e si trova al suo punto di consegna
        if has_package and (y, x) == self.target_delivery_points[drone_index]:
            self.deliveries_completed[drone_index] += 1  # aumenta solo per il drone specifico
            self.target_delivery_points[drone_index] = None  # reset del punto di consegna per il drone
            has_package = False

            # rimuove il punto di consegna dalla lista delle coordinate
            self.elements_coordinates['delivery_points'][drone_index] = None
        return has_package

    def __pick_up_package(self, y, x, has_package, drone_index):
        # verifica se il drone non ha un pacco e si trova nel magazzino
        if not has_package and (y, x) == self.WAREHOUSE and self.num_objects > 0:
            has_package = True
            self.num_objects -= 1

            # genera un nuovo punto di consegna casuale per il drone
            while True:
                new_delivery_point = (
                    np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
                )
                # verifica che il nuovo dp venga generato in una posizione che non è già occupata
                if new_delivery_point != self.WAREHOUSE and new_delivery_point not in self.CHARGING_STATIONS and new_delivery_point not in self.target_delivery_points and new_delivery_point not in self.elements_coordinates['drones']:
                    break

            self.target_delivery_points[drone_index] = new_delivery_point  # assegna il dp al drone specifico
            self.elements_coordinates['delivery_points'][drone_index] = new_delivery_point
        return has_package

    def __detect_obstacles(self, state, drone_index):
        y, x, battery_level, has_package = state
        obstacles = self.__get_obstacles(drone_index)

        obstacle_up = 1 if (y - 1, x) in obstacles else 0
        obstacle_down = 1 if (y + 1, x) in obstacles else 0
        obstacle_left = 1 if (y, x - 1) in obstacles else 0
        obstacle_right = 1 if (y, x + 1) in obstacles else 0

        return obstacle_up, obstacle_down, obstacle_left, obstacle_right

    def __determine_target(self, battery_level, has_package, drone_index):
        # non ci sono più consegne, il drone si dirige alla stazione di ricarica -> va in carica
        # oppure la batteria è scarica -> va in carica
        if (self.num_objects == 0 and not has_package) or battery_level < self.LOW_BATTERY_THRESHOLD:
            return self.CHARGING_STATIONS[drone_index]
        # ha un pacchetto -> deve andare al dp
        elif has_package and self.target_delivery_points[drone_index]:
            return self.target_delivery_points[drone_index]
        # deve andare al magazzino
        else:
            return self.WAREHOUSE

    def __check_collision(self, drone_index, new_y, new_x, y, x):
        # controlla se il drone collide con un ostacolo
        if (new_y, new_x) in self.__get_obstacles(drone_index):
            # print(f"Drone {drone_index} colliso con un ostacolo alle coordinate ({new_y}, {new_x})")
            return y, x  # ritorna le vecchie coordinate se c'è una collisione con un ostacolo

        # se non ci sono collisioni, ritorna le nuove coordinate
        return new_y, new_x

    def __generate_weather_zone(self):
        # genera una posizione casuale all'interno della griglia
        y = random.randint(0, self.grid_size[0] - 1)
        x = random.randint(0, self.grid_size[1] - 1)

        # genera dimensioni casuali per la zona
        width = random.randint(1, 3)  # larghezza casuale (da 1 a 3 celle)
        height = random.randint(1, 3)  # altezza casuale (da 1 a 3 celle)

        # aggiunge la nuova zona di maltempo alla lista con un timer di durata
        self.weather_zones.append({'position': (y, x), 'size': (width, height), 'lifetime': self.weather_lifetime})

    def __update_weather_zones(self):
        # decide casualmente se creare una nuova zona di maltempo
        if random.randint(0, self.weather_frequency) == 0:
            self.__generate_weather_zone()

        # aggiorna la durata delle zone esistenti e rimuove quelle scadute
        self.weather_zones = [zone for zone in self.weather_zones if zone['lifetime'] > 0]
        for zone in self.weather_zones:
            zone['lifetime'] -= 1

    def __detect_weather_zones(self, position, target_position, relative_direction):
        y, x = position
        tgt_y, tgt_x = target_position
        weather_cells_count = 0

        for weather_zone in self.weather_zones:
            (zone_y, zone_x), (width, height), _ = weather_zone.values()

            # il drone si sta muovendo verso l'alto
            if relative_direction == ActionType.UP.value:
                # rileva una zona di maltempo sopra di lui
                if zone_y <= y and (zone_x <= x <= zone_x + width):
                    weather_cells_count += 1
            # movimento verso il basso
            elif relative_direction == ActionType.DOWN.value:
                # rileva una zona di maltempo sotto di lui
                if zone_y + height >= y and (zone_x <= x <= zone_x + width):
                    weather_cells_count += 1
            # movimento a sx
            elif relative_direction == ActionType.LEFT.value:
                # rileva zona di maltempo a sx
                if zone_x <= x and (zone_y <= y <= zone_y + height):
                    weather_cells_count += 1
            # movimento a dx
            elif relative_direction == ActionType.RIGHT.value:
                # rileva maltempo a dx
                if zone_x + width >= x and (zone_y <= y <= zone_y + height):
                    weather_cells_count += 1

            # stima se deve attraversare più di una cella all'interno della zona di maltempo
            if weather_cells_count > 0:
                # itera lungo la direzione del movimento per vedere se deve attraversare più celle con maltempo
                # il movimento è 1 se sta muovendo in alto o verso dx, -1 se muove verso il basso o sx
                step = 1 if relative_direction in [ActionType.UP.value, ActionType.RIGHT.value] else -1

                # scorre le celle lungo la direzione del drone
                for i in range(1, abs(tgt_y - y) + 1) if relative_direction in [ActionType.UP.value,
                                                                                ActionType.DOWN.value] else range(1,
                                                                                                                  abs(tgt_x - x) + 1):

                    # calcola la prossima cella lungo la traiettoria:
                    # se il drone si muove in verticale aggiorna la y
                    # se il drone si muove in orizzontale aggiorna la x
                    next_y = y + i * step if relative_direction in [ActionType.UP.value, ActionType.DOWN.value] else y
                    next_x = x + i * step if relative_direction in [ActionType.LEFT.value, ActionType.RIGHT.value] else x

                    # verifica se la prossima cella da attraversare si trova all'interno di una zona di maltempo
                    if (zone_x <= next_x <= zone_x + width) and (zone_y <= next_y <= zone_y + height):
                        weather_cells_count += 1
                        if weather_cells_count >= 2:
                            return True # ritorna True solo se attraversa almeno due celle consecutive di maltempo
                    else:
                        break # se non è più in una zona di maltempo interrompe il controllo

        return False # se attraverso meno di due celle di maltempo consecutive ritorna False

    def __calculate_weather_circumnavigation_path(self, start_position, target_position, drone_index):
        # recupera gli ostacoli fisici
        obstacles = set(self.__get_obstacles(drone_index))

        # recupera le zone di maltempo attive
        weather_zones = set()
        for weather_zone in self.weather_zones:
            (zone_y, zone_x), (width, height), _ = weather_zone.values()
            # aggiunge tutte le celle che compongono le zone di maltempo
            for y in range(zone_y, zone_y + height):
                for x in range(zone_x, zone_x + width):
                    weather_zones.add((y, x))

        # unisce ostacoli fisici e zone di maltempo
        combined_obstacles = obstacles.union(weather_zones)

        # crea una coda a doppia estremità e la inizializza con la posizione di partenza e un percorso vuoto
        # ogni elemento della coda è una tupla che contiene la posizione corrente e il percorso esplorato
        queue = deque([(start_position, [])])

        # set per memorizzare le celle già visitate
        visited = set()
        visited.add(start_position)

        # direzioni possibili
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # finché la coda non è vuota continua a esplorare il percorso
        while queue:
            # estrae il primo elemento della coda, posizione corrente e percorso associato
            (current_position, path) = queue.popleft()
            y, x = current_position

            # se ha raggiunto il target restituisce il percorso
            if current_position == target_position:
                return path

            # per ogni direzione possibile calcola la nuova posizione
            for dy, dx in directions:
                new_y, new_x = y + dy, x + dx
                new_position = (new_y, new_x)

                # verifica che la nuova posizione sia valida e non ricada tra gli ostacoli
                if (0 <= new_y < self.grid_size[0] and 0 <= new_x < self.grid_size[1] and
                        new_position not in combined_obstacles and new_position not in visited):
                    visited.add(new_position)
                    queue.append((new_position, path + [new_position]))

        # se non c'è un percorso sicuro, attraversa comunque la zona di maltempo
        return self.__calculate_circumnavigation_path(start_position, target_position, drone_index)

    # decrementa la batteria del drone se la sua posizione è in una zona con maltempo attivo
    def __decrement_battery_due_to_weather(self, y, x, battery_level):
        # controlla se la posizione del drone è in una zona di maltempo
        for zone in self.weather_zones:
            zone_x, zone_y = zone['position']
            width, height = zone['size']

            # verifica se la posizione del drone è all'interno della zona di maltempo
            if zone_x <= x < zone_x + width and zone_y <= y < zone_y + height:
                battery_level = max(0, battery_level - 1)
                break # esce dal ciclo se la batteria è stata decrementata

        return battery_level



