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
    print("iteracje rosnąco: " + str({iteration_count}))
    return lista

def bubble_sort_desc(lista_desc):
    iteration_count_desc = 0
    n = len(lista_desc)
    for i in range(n):
        iteration_count_desc += 1
        swapped = False
        for j in range(0, n - i - 1):
            if lista_desc[j] < lista_desc[j + 1]:
                swapped = True
                lista_desc[j + 1], lista_desc[j] = lista_desc[j], lista_desc[j + 1]
                if not swapped:
                    break
    print("iteracje malejąco: " + str({iteration_count_desc}))
    return lista_desc
    
   

sorted_asc = bubble_sort_asc(NIU.copy())
sorted_desc = bubble_sort_desc(NIU.copy())

print("Lista przed sortowaniem:")
print(NIU)
print("lista po sortowaniu malejąco:")
print(sorted_asc)
print("lista po sortowaniu rosnąco:")
print(sorted_desc)