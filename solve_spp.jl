# Chargement des packages nécessaires
using JuMP, GLPK
using LinearAlgebra
using Printf  
include("loadSPP.jl")

# fonction pour créer le modèle SPP
function setSPP(C, A)
    m, n = size(A)
    spp = Model()
    @variable(spp, x[1:n], Bin)
    @objective(spp, Max, dot(C, x))
    @constraint(spp, cte[i=1:m], sum(A[i,j] * x[j] for j=1:n) <= 1)
    return spp
end

# Fonction pour résoudre le modèle et collecter les résultats
function solveSPPAndCollectResults(fname)
    # Chargement de l'instance SPP depuis un fichier
    C, A = loadSPP(fname)

    # Création du modèle SPP
    spp = setSPP(C, A)

    # Sélection du solveur GLPK pour résoudre le modèle
    set_optimizer(spp, GLPK.Optimizer)

    # Résolution du modèle et mesure du temps CPU
    time_start = time()
    optimize!(spp)
    elapsed_time = time() - time_start
    
    # Récupération de la valeur de l'objectif z
    z = objective_value(spp)
    return z, elapsed_time
end

# Fonction pour effectuer une recherche locale sur la solution
function localSearch(spp, C, A, initial_solution)
    n = length(initial_solution)
    current_solution = copy(initial_solution)
    current_value = dot(C, current_solution)
    
    # Boucle de recherche locale
    while true
        improved = false
        
        # Parcourez chaque ensemble
        for j in 1:n
            # Inversez l'état de l'ensemble j
            current_solution[j] = 1 - current_solution[j]
            
            # Vérifiez si la nouvelle solution est meilleure
            new_value = dot(C, current_solution)
            if new_value > current_value
                current_value = new_value
                improved = true
            else
                # Revert to the previous state if not improved
                current_solution[j] = 1 - current_solution[j]
            end
        end
        
        # Sortez de la boucle si aucune amélioration n'est trouvée
        if !improved
            break
        end
    end
    
    return current_solution, current_value
end

# Liste des noms de fichiers d'instances de test (adaptez ces noms et chemins)
instance_files = [
    "instance1.dat",
    "instance2.dat",
    # Ajoutez ici les noms des autres fichiers d'instances
]

# Boucle sur les instances de test pour résoudre avec heuristique
results_heuristic = Dict()
for (idx, instance_file) in enumerate(instance_files)
    println("Instance $idx: $instance_file")

    # Résoudre le modèle SPP avec l'heuristique de construction
    z_heuristic, elapsed_time_heuristic = solveSPPAndCollectResults(instance_file)

    # Afficher les résultats de l'heuristique
    @printf("Heuristique - z = %.2f\n", z_heuristic)
    @printf("Temps CPU (s) : %.2f\n", elapsed_time_heuristic)

    # Appliquer la recherche locale pour améliorer la solution
    C, A = loadSPP(instance_file)
    spp = setSPP(C, A)
    set_optimizer(spp, GLPK.Optimizer)
    # Résoudre le modèle initial
    optimize!(spp)
    initial_solution = value.(spp[:x])  # Utiliser la solution de l'heuristique comme point de départ
    improved_solution, improved_value = localSearch(spp, C, A, initial_solution)

    # Appelez optimize! à nouveau pour mettre à jour la solution après la recherche locale
    optimize!(spp)
    # Afficher les résultats de la recherche locale
    @printf("Après recherche locale - z = %.2f\n", improved_value)
    @printf("Temps CPU (s) : %.2f\n", elapsed_time_heuristic)

    # Enregistrer les résultats de la recherche locale
    results_heuristic[idx] = (z = improved_value, cpu_time = elapsed_time_heuristic)
end

# Afficher les résultats finaux
println("Résultats finaux - Heuristique :")
for idx in 1:length(instance_files)
    @printf("Instance %d - Heuristique - z = %.2f, Temps CPU (s) : %.2f\n", idx, results_heuristic[idx].z, results_heuristic[idx].cpu_time)
end

