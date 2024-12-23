import numpy as np
from typing import List, Tuple
import random
from PIL import Image


class MapGenerator:
    MOUNTAIN = 0
    RIVER = 1
    GRASS = 2
    ROCK = 3
    RIVERROCK = 4

    def __init__(self, width=50, height=50, population_size=100):
        self.width = width
        self.height = height
        self.population_size = population_size

    def generate_initial_population(self) -> List[np.ndarray]:
        population = []
        for _ in range(self.population_size):
            map_array = np.full((self.height, self.width), self.GRASS)
            for i in range(self.height):
                for j in range(self.width):
                    rand = np.random.random()
                    if rand < 0.02:
                        map_array[i, j] = self.ROCK
                    elif rand < 0.04:
                        map_array[i, j] = self.RIVERROCK
                    elif rand < 0.40:
                        map_array[i, j] = self.MOUNTAIN
                    elif rand < 0.70:
                        map_array[i, j] = self.RIVER
                    else:
                        map_array[i, j] = self.GRASS

            # 生成山脈（左右兩側）
            # mountain_width = self.width // 6
            # for y in range(self.height):
            #     # 修改計算方式，使上寬下窄
            #     current_width = int(mountain_width + ((self.height - y) / self.height) * mountain_width)
            #
            #     # 左側山脈
            #     map_array[y, :current_width] = self.MOUNTAIN
            #     # 右側山脈
            #     map_array[y, -current_width:] = self.MOUNTAIN
            #
            # # 生成主河道
            # river_x = self.width // 2
            # for y in range(self.height):
            #     map_array[y, river_x] = self.RIVER
            #     # 隨機偏移
            #     river_x += random.choice([-1, 0, 1])
            #     river_x = max(mountain_width, min(self.width - mountain_width - 1, river_x))
            #
            # # 添加支流
            # self._add_tributaries(map_array)
            #
            # # 添加石頭
            # self._add_rocks(map_array)
            #
            # # 添加河邊石頭
            # self._add_riverrocks(map_array)

            population.append(map_array)

        return population

    def fitness(self, map_array: np.ndarray) -> float:
        score = 0

        # 評估山脈分布
        mountain_score = self._evaluate_mountain_distribution(map_array)

        # 評估河流的連續性和流向
        river_score = self._evaluate_river_flow(map_array)

        # 評估石頭分布
        rock_score = self._evaluate_rock_distribution(map_array)

        # 評估河邊石頭
        riverrock_score = self._evaluate_riverrock_placement(map_array)

        # 評估草地的V形分布
        grass_score = self._evaluate_grass_distribution(map_array)

        return mountain_score + river_score + rock_score + riverrock_score + grass_score

    def _evaluate_mountain_distribution(self, map_array: np.ndarray) -> float:
        score = 0
        for y in range(self.height):
            for x in range(self.width):
                if map_array[y, x] == self.MOUNTAIN:
                    # 上半部分山脈可以較寬
                    if y < self.height // 2:
                        ideal_dis_away_center = self.width // 5
                    elif y < self.height // 1.5:
                        ideal_dis_away_center = self.width // 3
                    else:
                        ideal_dis_away_center = self.width // 2.1
                    # 計算到中心的距離
                    distance_to_center = abs(x - self.width // 2)
                    if distance_to_center < ideal_dis_away_center:
                        score -= ideal_dis_away_center - distance_to_center
                    else:
                        score += (distance_to_center - ideal_dis_away_center)

        return score

    def _check_river_connectivity(self, map_array: np.ndarray) -> Tuple[bool, int]:
        # 找出所有河流位置
        river_positions = np.where(map_array == self.RIVER)
        if len(river_positions[0]) == 0:
            return False, 0

        # 建立訪問記錄
        visited = np.zeros_like(map_array, dtype=bool)
        components = 0

        def dfs(y, x):
            if (y < 0 or y >= self.height or x < 0 or x >= self.width or
                    visited[y, x] or (map_array[y, x] != self.RIVER and map_array[y, x] != self.RIVERROCK)):
                return

            visited[y, x] = True
            # 檢查4個方向
            dfs(y - 1, x)
            dfs(y + 1, x)
            dfs(y, x - 1)
            dfs(y, x + 1)

        # 進行連通分量檢查
        for y, x in zip(*river_positions):
            if not visited[y, x]:
                components += 1
                dfs(y, x)

        return components == 1, components

    def _evaluate_river_flow(self, map_array: np.ndarray) -> float:
        score = 0
        river_positions = np.where(map_array == self.RIVER)

        # 檢查連通性
        is_connected, num_components = self._check_river_connectivity(map_array)
        if not is_connected:
            score -= 10 * num_components  # 嚴重懲罰不連通的情況

        # 計算上下半部分的河流密度
        upper_half = map_array[:self.height // 2]
        lower_half = map_array[self.height // 2:]

        upper_density = np.sum(upper_half == self.RIVER) / upper_half.size
        lower_density = np.sum(lower_half == self.RIVER) / lower_half.size

        # 獎勵上窄下寬的分布
        if upper_density < lower_density:
            score += 50

        # 評估每個河流格子
        for y, x in zip(*river_positions):
            # 根據位置給予不同的獎勵/懲罰
            if y < self.height // 2:
                # 上半部分：獎勵集中
                distance_to_center = abs(x - self.width // 2)
                if distance_to_center > 2:
                    score -= 20
            else:
                # 下半部分：限制河流寬度為1
                neighbors = 0
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-neighbors
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width and (
                            map_array[ny, nx] == self.RIVER or map_array[ny, nx] == self.RIVERROCK):
                        neighbors += 1
                if neighbors > 2:  # 如果河流寬度超過 1，懲罰
                    score -= 10
                else:
                    score += 5  # 基礎獎勵

        return score

    def _evaluate_rock_distribution(self, map_array: np.ndarray) -> float:
        score = 0
        rock_positions = np.where(map_array == self.ROCK)
        for y, x in zip(*rock_positions):
            has_mountain_neighbor = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if (0 <= y + dy < self.height and 0 <= x + dx < self.width and
                            map_array[y + dy, x + dx] == self.MOUNTAIN):
                        has_mountain_neighbor = True
                        break
            if has_mountain_neighbor:
                score -= 10
        return score

    def _evaluate_riverrock_placement(self, map_array: np.ndarray) -> float:
        score = 0
        riverrock_positions = np.where(map_array == self.RIVERROCK)
        # riverrock_count = np.sum(map_array == self.RIVERROCK).item()
        # if riverrock_count > self.height * self.width / 100:
        #     score -= (riverrock_count - self.height * self.width // 100) * 10

        for y, x in zip(*riverrock_positions):
            # 檢查是否靠近河流
            has_river_neighbor = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if (0 <= y + dy < self.height and
                            0 <= x + dx < self.width and
                            map_array[y + dy, x + dx] == self.RIVER):
                        has_river_neighbor = True
                        # score += 5
                        break
            if not has_river_neighbor:
                score -= 10

        return score

    def _evaluate_grass_distribution(self, map_array: np.ndarray) -> float:
        score = 0
        for y in range(self.height):
            for x in range(self.width):
                if map_array[y, x] == self.GRASS:
                    if x < self.width // 2:
                        distance_to_edge = x
                    else:
                        distance_to_edge = self.width - x
                    if y < self.height // 2:
                        ideal_dis_away_edge = self.width // 2.2
                    elif y < self.height // 1.5:
                        ideal_dis_away_edge = self.width // 3
                    else:
                        ideal_dis_away_edge = self.width // 6

                    if distance_to_edge < ideal_dis_away_edge:
                        score -= ideal_dis_away_edge - distance_to_edge
                    else:
                        score += (distance_to_edge - ideal_dis_away_edge)
                    # if y < self.height // 2:
                    #     score -= distance_to_center
                    # else:
                    #     score -= distance_to_center // 4

        return score

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        child1 = parent1.copy()
        child2 = parent2.copy()

        # 多點交叉
        for _ in range(random.randint(1, 3)):  # 1-3次交換
            cross_y = random.randint(0, self.height - 1)
            cross_x = random.randint(0, self.width - 1)
            child1[cross_y:, cross_x:], child2[cross_y:, cross_x:] = child2[cross_y:, cross_x:].copy(), child1[cross_y:,
                                                                                                        cross_x:].copy()

        return child1, child2

    def swap_mutate(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(self.height * self.width // 25):
            # Swap mutation for child1
            y1, x1 = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            y2, x2 = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            child1[y1, x1], child1[y2, x2] = child1[y2, x2], child1[y1, x1]

        for i in range(self.height * self.width // 25):
            # Swap mutation for child2
            y1, x1 = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            y2, x2 = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            child2[y1, x1], child2[y2, x2] = child2[y2, x2], child2[y1, x1]

        return child1, child2

    def inversion_mutate(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Inversion mutation for child1
        # for _ in range(self.height * self.width // 25):  # 控制變異次數
        y1, y2 = sorted([random.randint(0, self.height - 1) for _ in range(2)])
        x1, x2 = sorted([random.randint(0, self.width - 1) for _ in range(2)])
        child1[y1:y2 + 1, x1:x2 + 1] = np.flip(child1[y1:y2 + 1, x1:x2 + 1], axis=None)

        # Inversion mutation for child2
        # for _ in range(self.height * self.width // 25):  # 控制變異次數
        y1, y2 = sorted([random.randint(0, self.height - 1) for _ in range(2)])
        x1, x2 = sorted([random.randint(0, self.width - 1) for _ in range(2)])
        child2[y1:y2 + 1, x1:x2 + 1] = np.flip(child2[y1:y2 + 1, x1:x2 + 1], axis=None)

        return child1, child2

    def tournament_select(self, population: List[np.ndarray], tournament_size: int = 3) -> np.ndarray:
        # 隨機選擇 tournament_size 個個體
        tournament = random.sample(population, tournament_size)
        # 計算這些個體的適應度
        tournament_fitness = [self.fitness(map_array) for map_array in tournament]
        # 返回適應度最高的個體
        return tournament[np.argmax(tournament_fitness)]

    def evolve(self, generations: int = 100) -> np.ndarray:
        # 初始化種群
        population = self.generate_initial_population()
        generation = 0
        while True:
            # for generation in range(generations):
            # 計算適應度
            fitness_scores = [self.fitness(map_array) for map_array in population]

            # 選擇最佳個體
            elite_size = self.population_size // 10
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            new_population = [population[i].copy() for i in elite_indices]
            # 生成新的種群
            while len(new_population) < self.population_size:
                # 選擇父母
                # parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
                parent1 = self.tournament_select(population)
                parent2 = self.tournament_select(population)

                if random.random() < 0.9:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = self.inversion_mutate(parent1, parent2)

                new_population.extend([child1, child2])

            # 更新種群
            population = new_population[:self.population_size]

            if generation % 10 == 0:
                best_index = np.argmax([self.fitness(map_array) for map_array in population])
                create_map_image(population[best_index], save_path=f"GAoutput/{generation}.png")
                print(f"Generation {generation}, Best Fitness: {max(fitness_scores)}")
                if generation > 10 and abs(max(fitness_scores) - old_fit) < 25:
                    break
                old_fit = max(fitness_scores)
            generation += 1

        # 返回最佳地圖
        best_index = np.argmax([self.fitness(map_array) for map_array in population])
        return population[best_index]


def create_map_image(map_array, tile_size=(64, 64), save_path: str = None):
    # 定義圖塊對應關係
    tile_map = {
        0: "data/mountain.png",  # 山脈/灰色區域
        1: "data/river.png",  # 河流/藍色區域
        2: "data/grass.png",  # 草地/綠色區域
        3: "data/rock.png",  # 岩石
        4: "data/riverstone.png"  # 河邊的石頭
    }

    # 載入所有圖塊
    tiles = {}
    for key, filename in tile_map.items():
        try:
            tiles[key] = Image.open(filename).resize(tile_size)
        except FileNotFoundError:
            # 如果找不到圖檔，創建對應顏色的方塊
            color = {
                0: (128, 128, 128),  # 灰色代表山脈
                1: (0, 128, 255),  # 藍色代表河流
                2: (0, 255, 0),  # 綠色代表草地
                3: (169, 169, 169),  # 深灰色代表岩石
                4: (100, 100, 100)  # 淺灰色代表河邊石頭
            }.get(key, (0, 0, 0))
            tiles[key] = Image.new('RGB', tile_size, color)

    # 創建完整地圖圖像
    height, width = map_array.shape
    full_image = Image.new('RGB', (width * tile_size[0], height * tile_size[1]))

    # 填充地圖
    for y in range(height):
        for x in range(width):
            tile_type = map_array[y, x]
            if tile_type in tiles:
                full_image.paste(tiles[tile_type], (x * tile_size[0], y * tile_size[1]))

    if save_path:
        image = full_image.resize((width * 10, height * 10), Image.NEAREST)
        image.save(save_path)

    return full_image


# 使用示例
if __name__ == "__main__":
    generator = MapGenerator(width=40, height=45, population_size=200)
    for i in range(4, 11):
        best_map = generator.evolve(generations=180)
        create_map_image(best_map, save_path="output/map{}.png".format(i))
