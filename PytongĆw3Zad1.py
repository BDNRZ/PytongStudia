NIU = [8,4,2,1]

def bubble_sort_asc(lista):
    iteration_count = 0
    n = len(lista)
    for i in range(n):
        iteration_count += 1
        swapped = False
        for j in range(0, n - i - 1):
            if lista[j] > lista[j + 1]:
                swapped = True
                lista[j], lista[j + 1] = lista[j + 1], lista[j]
                if not swapped:
                    break

def bubble_sort_desc(lista):
    iteration_count = 0
    n = len(lista)
    for i in range(n):
        iteration_count += 1
        swapped = False
        for j in range(0, n - i - 1):
            if lista[j] < lista[j + 1]:
                swapped = True
                lista[j + 1], lista[j] = lista[j], lista[j + 1]
                if not swapped:
                    break
    
    print({iteration_count})
    return lista

sorted_asc = bubble_sort_asc(NIU.copy())
sorted_desc = bubble_sort_desc(NIU.copy())

print("Lista przed sortowaniem:")
print(NIU)
print("lista po sortowaniu:")
print(sorted_asc)
print("print desc:")
print(sorted_desc)