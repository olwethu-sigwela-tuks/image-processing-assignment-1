import numpy as np
import cv2
import random

np.seterr(all='raise')
random.seed(42)

def gen_initial_population(n, num_thresholds):
    
    population = []

    for i in range(n):
        thresholds = []
        for j in range(num_thresholds):
            thresholds.append(random.randint(0,255))
        thresholds.sort()
        population.append(thresholds)

    return population

def get_random_indices(curr_index, population):
    indices = []
    for i in range(3):
        random_index = curr_index
        while (random_index == curr_index): #none of the chosen indices can be the same as the current index
            choice = random.randint(0, len(population) - 1)
            if choice in indices: #we do not want the value we pick to be the same as any of the other chosen indices
                continue
            random_index = choice

        indices.append(random_index)
    return indices

def do_multilevel_thresholding(image, thresholds):

    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_intensities = np.array([round(i*(255/(len(thresholds) + 1))) for i in range(len(thresholds) + 1)], dtype=np.uint8)
    indices = np.digitize(grayscale, thresholds)
    thresholded = output_intensities[indices]

    return thresholded




def otsu_variance(histogram_normalized, thresholds):
    bounds = [0] + sorted([int(round(t)) for t in thresholds]) + [256]

    global_mean_intensity = sum(i * histogram_normalized[i] for i in range(256))

    total_variance = 0.0

    for j in range(len(bounds) - 1):
        lower = bounds[j]
        upper = bounds[j + 1]

        class_probability = np.sum(histogram_normalized[lower:upper])
        if class_probability == 0:
            continue

        class_mean_intensity = sum(i * histogram_normalized[i] for i in range(lower, upper)) / class_probability
        total_variance += class_probability * ((class_mean_intensity - global_mean_intensity)**2)

    return total_variance

def kapur_entropy(histogram_normalized, thresholds):
    bounds = [0] + sorted([int(round(t)) for t in thresholds]) + [256]

    total_entropy = 0.0

    for j in range(len(bounds) - 1):

        lower = bounds[j]
        upper = bounds[j + 1]

        class_probability = np.sum(histogram_normalized[lower:upper])

        if class_probability == 0:
            continue

        for i in range(lower,upper):
            p_i = histogram_normalized[i]
            if p_i > 0:
                p_normalized = p_i / class_probability
                total_entropy -= p_normalized*np.log(p_normalized)

    return total_entropy

def tsallis_entropy(histogram_normalized, thresholds, q=0.8):

    bounds = [0] + sorted([int(round(t)) for t in thresholds]) + [256]

    class_entropies = []

    for j in range(len(bounds) - 1):
        lower = bounds[j]
        upper = bounds[j+1]

        cumulative_probabilty = np.sum(histogram_normalized[lower:upper])

        if cumulative_probabilty == 0:
            continue

        class_entropy_fragment = 0
        for i in range(lower, upper):
            p_normalized = histogram_normalized[i]/cumulative_probabilty

            class_entropy_fragment += p_normalized**q

        class_entropy = (1/(q-1)) * (1 - class_entropy_fragment)
        class_entropies.append(class_entropy)

    if not class_entropies:
        return 0.0

    
    product = 1.0

    for s in class_entropies:
        product *= (1.0 + (1.0 - q) * s)

    total_entropy = (product - 1.0) / (1.0 - q)

    return total_entropy













def de(histogram, NP, num_thresholds, n_generations, F = 1, CR=0.5, objective=kapur_entropy):
    #NP = population size
    population =  gen_initial_population(NP, num_thresholds)
    rng = np.random.default_rng()
    # image = cv2.imread(image_path)

    lower_bound = 0
    upper_bound = 255

    # print(f"{image=}")

    for n in range(n_generations):
        print(f"GENERATION {n}")
        print(f"population: {population}")
        new_population = []

        for i in range(len(population)):
            r1, r2, r3 = get_random_indices(i, population)

            v = [] #v is the mutated vector

            #element-wise vector addition - not using numpy addition because of the custom logic for boundary conditions
            for x in range(len(population[r1])): 
                v_component = population[r1][x] + F * (population[r2][x] - population[r3][x])

                # Boundary conditions, handled according to
                # JADE: Adaptive Differential Evolution with
                # Optional External Archive
                # but with rounding so that the values remain integers
                if v_component < lower_bound:
                    v_component = round((lower_bound + population[i][x])/2)
                if v_component > upper_bound:
                    v_component = round((upper_bound + population[i][x])/2)
                
                v.append(v_component)

            # v = population[r1] + F * (population[r2] - population[r3]) 


            u = []

            rand_idx = random.randint(0, len(v) - 1)
            for j in range(len(population[i])):
                rand_j = rng.uniform(low=0.0, high=1.0, size=None)

                if (rand_j <= CR) or (j == rand_idx):
                    # print(f"{u=}")
                    # print(f"{v=}")
                    # print(f"{rand_idx=}")
                    # print(f"{j=}")
                    u.append(v[j])
                elif (rand_j > CR) and (j != rand_idx):
                    u.append(population[i][j])
            
            #selection
            u.sort()
            # print(f"{otsu_variance(image, u)=}")
            # print(f"{otsu_variance(image, population[i])=}")
            # print(f"{otsu_variance(image, u)=}")
            # print(f"{otsu_variance(image, population[i])=}")
            if objective(histogram, u) > objective(histogram, population[i]):
                # for po in range(50):
                #     print(po*"&")
                # print("better")
                new_population.append(np.array(u))
            else:
                # print("not better")
                new_population.append(population[i])

        # print(f"{(population == new_population)=}")
        population = new_population
            

def sort_lists_in_parallel(list1, list2):
    sorted1, sorted2 = zip(*sorted(zip(list1, list2)))

    return list(sorted1), list(sorted2)

        
def jade(histogram, NP, num_thresholds, n_generations, c = 0.1, p=0.05, objective=kapur_entropy):
    #NP = population size
    population =  gen_initial_population(NP, num_thresholds)
    scores = [otsu_variance(histogram, threshold) for threshold in population]

    scores, population = sort_lists_in_parallel(scores, population)
    
    rng = np.random.default_rng()
    # image = cv2.imread(image_path)

    lower_bound = 0
    upper_bound = 255

    archive = []

    s_CR = [] #set of all successful CR's
    s_F = [] #set of all successful F's

    CR_mean = 0.5
    F_mean = 0.5

    for n in range(n_generations):
        print(f"GENERATION {n}")
        print(f"population: {population}")
        new_population = []
        new_scores = []

        s_CR = []
        s_F = []

        
        for i in range(len(population)):
            CR_i = rng.normal(loc=CR_mean, scale=0.1)
            if CR_i < 0:
                CR_i = 0

            if CR_i > 1:
                CR_i = 1

            F_i = F_mean + (0.1*rng.standard_cauchy())

            if F_i > 1:
                F_i = 1

            if F_i <= 0:
                while F_i <= 0:
                    F_i = F_mean + (0.1*rng.standard_cauchy())

            p_num = max(1, int(round(p * len(population))))
            p_best = population[len(population) - p_num:]

            threshold_i = population[i]
            threshold_best = random.choice(p_best)

            r1_idx = random.randint(0, len(population) - 1)
            threshold_r1 = population[r1_idx]
            if r1_idx == i:
                while r1_idx == i:
                    r1_idx = random.randint(0, len(population) - 1)
                    threshold_r1 = population[r1_idx]

            union = population + archive
            r2_idx = random.randint(0, len(union) - 1)
            threshold_r2 = union[r2_idx]
            if (r2_idx == r1_idx) or (r2_idx == i):
                while (r2_idx == r1_idx) or (r2_idx == i):
                    r2_idx = random.randint(0, len(union) - 1)
                    threshold_r2 = union[r2_idx]

            v = [] #v is the mutated vector

            #element-wise vector addition - not using numpy addition because of the custom logic for boundary conditions
            for x in range(len(threshold_i)): 
                v_component = threshold_i[x] + (F_i * (threshold_best[x] - threshold_i[x])) + (F_i * (threshold_r1[x] - threshold_r2[x]))

                # Boundary conditions, handled according to
                # JADE: Adaptive Differential Evolution with
                # Optional External Archive
                # but with rounding so that the values remain integers
                if v_component < lower_bound:
                    v_component = round((lower_bound + threshold_i[x])/2)
                if v_component > upper_bound:
                    v_component = round((upper_bound + threshold_i[x])/2)
                
                v.append(v_component)



            u = []

            rand_idx = random.randint(0, len(v) - 1)
            for j in range(len(population[i])):
                rand_j = rng.uniform(low=0.0, high=1.0, size=None)

                if (rand_j <= CR_i) or (j == rand_idx):
             
                    u.append(v[j])
                elif (rand_j > CR_i) and (j != rand_idx):
                    u.append(population[i][j])
            
            #selection
            u.sort()

            u_score = objective(histogram, u)
            parent_score = objective(histogram, population[i])
            if u_score > parent_score:
      
                new_population.append(u)
                new_scores.append(u_score)
                s_CR.append(CR_i)
                s_F.append(F_i)
                archive.append(population[i])
                if len(archive) > NP:
                    random_removal_idx = random.randint(0, len(archive) - 1)
                    archive.pop(random_removal_idx)
            else:
                new_population.append(population[i])
                new_scores.append(parent_score)
                

        population = new_population
        scores = new_scores
        scores, population = sort_lists_in_parallel(scores, population)
        if len(s_CR) > 0 and len(s_F) > 0:
            CR_mean = ((1 - c) * CR_mean) + (c * np.mean(s_CR))
            F_mean = ((1 - c) * F_mean) + (c * lehmer_mean(s_F))

    return population[-1] #returns best set of thresholds 
        
            
        
def lehmer_mean(lst):
    return sum(i**2 for i in lst)/sum(i for i in lst)



def main():
    image_path = "..\\BDS500\\BDS500\\img1.png"
    # image_path = "C:\\Users\\Olwethu\\Documents\\School\\COS 791\\Assignment 1\\BDS500\\BDS500\\img7.png" #TODO: NB - change this back to the relative path (only using absolute path so that the debugger works)
    
    image = cv2.imread(image_path)
    # thresholded = do_multilevel_thresholding(image, thresholds)
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram_normalized = cv2.normalize(histogram, None, alpha = 1, beta = 0, norm_type=cv2.NORM_L1)
    
    # de(histogram_normalized, 1000, 12, 100, 1, 0.5)
    jade(histogram=histogram_normalized, NP=50, num_thresholds=3, n_generations=100, c = 0.1, p = 0.05, objective=tsallis_entropy)
    
if __name__ == "__main__":
    main()





