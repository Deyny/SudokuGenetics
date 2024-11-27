import numpy as np
import random
from deap import base, creator, tools, algorithms

# Przykładowe plansze Sudoku o różnym poziomie trudności
easy = [
    [5, 3, 0, 0, 7, 0, 9, 0, 0],
    [6, 7, 2, 1, 9, 0, 3, 4, 8],
    [0, 0, 8, 0, 0, 0, 5, 6, 0],
    [8, 0, 9, 0, 6, 1, 0, 2, 0],
    [4, 0, 6, 8, 5, 0, 7, 0, 0],
    [7, 1, 3, 0, 0, 0, 8, 5, 6],
    [9, 0, 1, 5, 0, 7, 2, 8, 0],
    [2, 0, 0, 0, 1, 0, 6, 0, 5],
    [0, 4, 5, 0, 8, 0, 0, 7, 0]
]

hard = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 8],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

v_hard = [
    [5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 9, 0, 0, 4, 0],
    [0, 9, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 7, 0]
]

# Wybór planszy, którą chcemy rozwiązać
sudoku_board = np.array(easy)

# Tworzenie klas definiujących fitness i osobnika
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Fitness minimalizacyjny
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)  # Typ osobnika (tablica dwuwymiarowa)


# Funkcja tworząca osobników
def create_individual():
    # Tworzymy kopię planszy startowej
    individual = sudoku_board.copy()
    # Wypełniamy puste miejsca (wartości 0) losowymi liczbami od 1 do 9
    for i in range(9):
        row = individual[i]
        missing_numbers = list(set(range(1, 10)) - set(row))  # Znalezienie brakujących liczb w wierszu
        random.shuffle(missing_numbers)  # Losowe ułożenie brakujących liczb
        for j in range(9):
            if row[j] == 0:  # Wstawiamy brakujące liczby w miejsca zer
                row[j] = missing_numbers.pop()
    return individual


# Funkcja oceny (fitness) osobnika
def eval_sudoku(individual):
    fitness = 0

    # Ocena unikalności wartości w wierszach
    for row in individual:
        fitness += len(row) - len(set(row))  # Liczba powtórzeń w wierszu

    # Ocena unikalności wartości w kolumnach
    for col in individual.T:
        fitness += len(col) - len(set(col))  # Liczba powtórzeń w kolumnie

    # Ocena unikalności wartości w blokach
    for i in range(3):
        for j in range(3):
            block = individual[i*3:(i+1)*3, j*3:(j+1)*3].flatten()  # Pobranie bloku jako tablicy jednowymiarowej
            fitness += len(block) - len(set(block))  # Liczba powtórzeń w bloku

    return fitness,


# Funkcja mutacji osobnika
def mutate_individual(individual):
    # Znalezienie komórek, które można zmieniać (pustych w początkowej planszy)
    mutable_cells = [(i, j) for i in range(9) for j in range(9) if sudoku_board[i][j] == 0]

    # Zamiana miejscami dwóch losowych wartości
    if len(mutable_cells) >= 2:
        (x1, y1), (x2, y2) = random.sample(mutable_cells, 2)
        individual[x1, y1], individual[x2, y2] = individual[x2, y2], individual[x1, y1]

    return individual,


# Definiowanie narzędzi DEAP
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)  # Tworzenie osobnika
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Tworzenie populacji
toolbox.register("evaluate", eval_sudoku)  # Ocena osobników
toolbox.register("mate", tools.cxTwoPoint)  # Krzyżowanie osobników
toolbox.register("mutate", mutate_individual)  # Mutowanie osobników
toolbox.register("select", tools.selTournament, tournsize=7)  # Selekcja osobników


# Parametry algorytmu ewolucyjnego
population = toolbox.population(n=7000)  # Rozmiar populacji
max_generations = 100  # Maksymalna liczba generacji
crossover_probability = 0.7  # Prawdopodobieństwo krzyżowania
mutation_probability = 0.3  # Prawdopodobieństwo mutacji

elitism_size = 2  # Liczba elitarnych osobników
stagnations = 5  # Liczba generacji bez poprawy
mutation_tempo = 0.05  # Tempo wzrostu prawdopodobieństwa mutacji
max_mut_prob = 0.8  # Maksymalna wartość prawdopodobieństwa mutacji


# Ewaluacja początkowej populacji
for ind in population:
    ind.fitness.values = toolbox.evaluate(ind)

best_fitness_previous = 999  # Duża wartość na start
start_mut_prob = mutation_probability  # Początkowe prawdopodobieństwo mutacji
stagnation_counter = 0  # Licznik stagnacji


# Główna pętla algorytmu ewolucyjnego
for gen in range(max_generations):
    # Generowanie nowego potomstwa
    offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability)

    # Ewaluacja potomstwa
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit  # Przypisanie fitnessu potomkom

    # Zachowanie elit
    elites = tools.selBest(population, elitism_size)

    # Selekcja do nowej populacji
    selected = toolbox.select(offspring, k=len(population) - elitism_size)
    population = elites + selected  # Tworzenie nowej populacji

    # Obliczanie statystyk
    fits = [ind.fitness.values[0] for ind in population]
    best_fitness = min(fits)  # Najlepsza wartość fitness w tej generacji
    mean_fitness = sum(fits) / len(fits)  # Średnia wartość fitness
    print(f"Generacja {gen}: Najlepsze dopasowanie = {best_fitness}, Średnie dopasowanie = {mean_fitness}")

    # Sprawdzenie, czy znaleziono rozwiązanie
    best_ind = tools.selBest(population, 1)[0]
    if best_ind.fitness.values[0] == 0:  # Jeśli fitness wynosi 0, rozwiązanie jest poprawne
        print("Znaleziono rozwiązanie w generacji", gen)
        break

    # Mechanizm adaptacyjny dla mutacji
    if best_fitness == best_fitness_previous:
        stagnation_counter += 1  # Wzrost licznika stagnacji
    else:
        if best_fitness < best_fitness_previous:  # Lepsze rozwiązanie
            mutation_probability = start_mut_prob  # Resetowanie prawdopodobieństwa mutacji
        stagnation_counter = 0
        best_fitness_previous = best_fitness

    if stagnation_counter >= stagnations:  # Jeśli wystąpi stagnacja
        mutation_probability = min(mutation_probability + mutation_tempo, max_mut_prob)  # Zwiększanie mutacji
        print(f"Stagnacja od {stagnations} generacji. Zwiększam prawdopodobieństwo mutacji do {mutation_probability:.2f}")
        stagnation_counter = 0


# Wyświetlenie najlepszego rozwiązania
print("Najlepszy osobnik:")
print(best_ind)  # Ostateczna plansza Sudoku
print("Fitness:", best_ind.fitness.values[0])  # Ostateczny fitness
