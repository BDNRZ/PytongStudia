def graph_maker():
    len_a = int(input("Podaj wysokość pierwszego słupka (a): "))
    len_b = int(input("Podaj wysokość drugiego słupka (b): "))
    
    if len_a < 0 or len_a >= 9 or len_b < 0 or len_b >= 9:
        print("Wysokość słupków musi być większa lub równa 0 i mniejsza niż 9!")
        return
    
    if len_a == len_b:
        print("Wysokości słupków muszą być różne!")
        return
    
    len_c = abs(len_a - len_b)
    
    print(f"a = {len_a}, b = {len_b}, c = {len_c}")
    
    for i in range(max(len_a, len_b, len_c) + 1):
        if i == 0 or i == len_a:
            print("+", end="")
            print("-", end="")
            print("+", end="")
        elif i < len_a:
            print("|", end="")
            print(" ", end="")
            print("|", end="")
        else:
            print(" ", end="")
            print(" ", end="")
            print(" ", end="")
        
        print(" ", end="")
        
        if i == 0 or i == len_b:
            print("+", end="")
            print("-", end="")
            print("+", end="")
        elif i < len_b:
            print("|", end="")
            print(" ", end="")
            print("|", end="")
        else:
            print(" ", end="")
            print(" ", end="")
            print(" ", end="")
        
        print(" ", end="")
        
        if i == 0 or i == len_c:
            print("+", end="")
            print("-", end="")
            print("+", end="")
        elif i < len_c:
            print("|", end="")
            print(" ", end="")
            print("|", end="")
        else:
            print(" ", end="")
            print(" ", end="")
            print(" ", end="")
        
        print()
    
if __name__ == "__main__":
    graph_maker()