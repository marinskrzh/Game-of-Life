import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


MAX_AGE = 5  # Максимальное время жизни клетки
MUTATION_PROBABILITY = 0.01  # Вероятность появления мутанта
MUTATION_STRENGTH = 3  # Сила влияния мутанта на соседей



# Создаем случайную матрицу 50x50 с значениями 0 (мертвая) или 1 (живая)
def create_initial_grid(size=50):
    return np.random.choice([0, 1], size=(size, size))


# Создаем матрицу для отслеживания времени жизни клеток
def create_age_grid(size=50):
    return np.zeros((size, size), dtype=int)


# Создаем матрицу для отслеживания мутантов (0 - обычная, 1 - мутант)
def create_mutant_grid(size=50):
    return np.zeros((size, size), dtype=int)


def count_neighbors(grid):
    # Создаем ядро для свертки (соседи включая саму клетку)
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # исключаем саму клетку

    # Создаем матрицу для подсчета соседей у каждой клетки
    neighbors = np.zeros_like(grid, dtype=int)

    # Обработка границ(тороидальная вселенная)
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            # Сдвигаем сетку и добавляем к соседям
            neighbors += np.roll(np.roll(grid, i - 1, axis=0), j - 1, axis=1)

    return neighbors


def apply_mutation(grid, mutant_grid):
    size = grid.shape[0]
    new_grid = grid.copy()
    new_mutant_grid = mutant_grid.copy()
    changes = []  # Изменения для согласованного применения
    

    for i in range(size):
        for j in range(size):
            if random.random() < MUTATION_PROBABILITY: # Условие для мутации
                # Запоминаем мутацию и ее влияние
                changes.append(('mutant', i, j, random.choice([0, 1])))
                
                # Влияние на соседей
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = (i + di) % size, (j + dj) % size # Вычисление координат соседа
                        if random.random() < 0.3:
                            changes.append(('neighbor', ni, nj, random.choice([0, 1])))
    
    # Согласованное применение изменений
    for change_type, i, j, new_state in changes:
        if change_type == 'mutant':
            new_grid[i, j] = new_state
            new_mutant_grid[i, j] = 1
        else:
            current_neighbors = count_neighbors(grid)[i, j]
            if (new_state == 1 and current_neighbors == 3) or \
               (new_state == 0 and (current_neighbors < 2 or current_neighbors > 3)):
                new_grid[i, j] = new_state
    
    return new_grid, new_mutant_grid




def update_grid(grid, age_grid, mutant_grid):


    neighbors = count_neighbors(grid)

    # Правила для Игры
    die_underpopulated = (neighbors < 2)
    survive = (neighbors == 2) | (neighbors == 3)
    die_overpopulated = (neighbors > 3)
    become_alive = (neighbors == 3)
    die_old_age = (age_grid >= MAX_AGE)

    die_mask = die_underpopulated | die_overpopulated | die_old_age
    
    new_grid = grid.copy()
    
    # Обновляем сетки
    new_grid[die_mask & (grid == 1)] = 0 
    new_grid[survive & (grid == 1)] = 1
    new_grid[become_alive & (grid == 0)] = 1

    
    new_mutant_grid = mutant_grid.copy()
    new_mutant_grid[die_mask & (mutant_grid == 1)] = 0
    new_mutant_grid[become_alive & (grid == 0)] = 0

    # Обновляем возраст клеток
    new_age_grid = age_grid.copy()
    new_age_grid[(new_grid == 1) & (grid == 1)] += 1
    new_age_grid[(new_grid == 1) & (grid == 0)] = 0
    new_age_grid[new_grid == 0] = 0

    # Применяем мутации к обновленной сетке
    new_grid, new_mutant_grid = apply_mutation(new_grid, new_mutant_grid)

    return new_grid, new_age_grid, new_mutant_grid

# Инициализируем сетки
grid = create_initial_grid()
age_grid = create_age_grid()
mutant_grid = create_mutant_grid()

# Создаем фигуру и оси
fig, ax = plt.subplots(figsize=(8, 8))

# Настраиваем отображение (черный - мертвая, белый - живая)
img = ax.imshow(grid, cmap='binary', interpolation='nearest')


# Добавляем отображение мутантов поверх основной сетки
mutant_img = ax.imshow(np.ma.masked_where(mutant_grid == 0, mutant_grid), alpha=0.5, interpolation='nearest')

ax.set_title('Игра "Жизнь" - Поколение 0')
ax.axis('off')
plt.tight_layout()


def update_frame(frame):
    global grid, age_grid, mutant_grid

    # Обновляем сетку
    grid, age_grid, mutant_grid = update_grid(grid, age_grid, mutant_grid)
    
    # Обновляем изображение
    img.set_array(grid)
    mutant_img.set_array(np.ma.masked_where(mutant_grid == 0, mutant_grid))

    # Обновляем заголовок
    ax.set_title(f'Поколение {frame} (Мутанты: {np.sum(mutant_grid)})')

    return [img, mutant_img, ax.title]


# Создаем анимацию
ani = animation.FuncAnimation(
    fig,
    update_frame,
    frames=100,
    interval=200,
)

# Показ анимации
plt.show()